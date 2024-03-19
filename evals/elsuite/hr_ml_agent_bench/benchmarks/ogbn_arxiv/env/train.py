import pandas as pd
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

target_dataset = "ogbn-arxiv"

dataset = PygNodePropPredDataset(name=target_dataset, root="networks")
data = dataset[0]
split_idx = dataset.get_idx_split()

train_idx = split_idx["train"]
valid_idx = split_idx["valid"]
test_idx = split_idx["test"]

train_loader = NeighborLoader(
    data,
    input_nodes=train_idx,
    shuffle=True,
    num_workers=1,
    batch_size=32,
    num_neighbors=[30] * 2,
)

total_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=32,
    shuffle=False,
    num_workers=1,
)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)

    def inference(self, total_loader, device):
        xs = []
        for batch in total_loader:
            out = self.forward(batch.x.to(device))
            out = out[: batch.batch_size]
            xs.append(out.cpu())

        out_all = torch.cat(xs, dim=0)

        return out_all


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = SAGE(data.x.shape[1], 256, dataset.num_classes, n_layers=2)
model = MLP(data.x.size(-1), hidden_channels=16, out_channels=172, num_layers=2, dropout=0).to(
    device
)

model.to(device)
epochs = 4
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = ReduceLROnPlateau(optimizer, "max", patience=7)


def test(model, device):
    evaluator = Evaluator(name=target_dataset)
    model.eval()
    out = model.inference(total_loader, device)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, val_acc, test_acc


for epoch in range(epochs):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0

    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()

        out = model(batch.x.to(device))
        out = out[:batch_size]

        batch_y = batch.y[:batch_size].to(device)
        batch_y = torch.reshape(batch_y, (-1,))

        loss = F.nll_loss(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    train_acc, val_acc, test_acc = test(model, device)

    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}")

evaluator = Evaluator(name=target_dataset)
model.eval()
out = model.inference(total_loader, device)
y_pred = out.argmax(dim=-1, keepdim=True)

y_pred_np = y_pred[split_idx["test"]].numpy()
df = pd.DataFrame(y_pred_np)
df.to_csv("submission.csv", index=False)

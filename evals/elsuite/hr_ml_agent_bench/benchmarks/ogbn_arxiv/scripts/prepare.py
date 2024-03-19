from pathlib import Path

from ogb.nodeproppred import PygNodePropPredDataset

env_dir = Path(__file__).parent / ".." / "env"
dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=env_dir / "networks")

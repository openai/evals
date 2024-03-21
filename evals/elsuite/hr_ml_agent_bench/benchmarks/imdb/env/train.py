import pandas as pd
import torch
from datasets import load_dataset

if __name__ == "__main__":
    imdb = load_dataset("imdb")

    # TODO: pre-process data

    model: torch.nn.Module = None  # TODO: define model here

    # TODO: train model

    ############################################
    #                                          #
    #  Do not modify anything below this line! #
    #                                          #
    ############################################

    # Set model to evaluation mode
    model.eval()

    # Evaluate the model on the test set and save the predictions to submission.csv.
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    n_correct = 0

    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        y_true = data["label"]

        with torch.no_grad():
            logits = model(text)

        logits = torch.softmax(logits, dim=0)
        y_pred = torch.argmax(logits).item()
        n_correct += int(y_pred == y_true)

        submission.loc[idx] = logits.tolist()

    accuracy = 100.0 * n_correct / len(imdb["test"])
    submission.to_csv("submission.csv", index_label="idx")

    print("Accuracy: ", accuracy)

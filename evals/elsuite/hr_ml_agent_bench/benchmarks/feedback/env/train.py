import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

DIMENSIONS = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [y[DIMENSIONS.index(task)] for y in y_test]
        pred_task = [y[DIMENSIONS.index(task)] for y in y_test_pred]
        rmse = mean_squared_error(targets_task, pred_task, squared=False)
        metrics[f"rmse_{task}"] = rmse

    return metrics


def train_model(X_train, y_train, X_valid, y_valid):
    # TODO. define and train the model
    # should return the trained model
    model = None
    return model


def predict(model, X):
    # TODO. predict the model
    # should return an array of predictions
    y_pred = np.random.rand(len(X), len(DIMENSIONS))
    return y_pred


if __name__ == "__main__":
    ellipse_df = pd.read_csv(
        "train.csv",
        header=0,
        names=[
            "text_id",
            "full_text",
            "Cohesion",
            "Syntax",
            "Vocabulary",
            "Phraseology",
            "Grammar",
            "Conventions",
        ],
        index_col="text_id",
    )
    ellipse_df = ellipse_df.dropna(axis=0)

    # Process data and store into numpy arrays.
    data_df = ellipse_df
    X = list(data_df.full_text.to_numpy())
    y = np.array([data_df.drop(["full_text"], axis=1).iloc[i] for i in range(len(X))])

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    # should fill out the train_model function
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    # should fill out the predict function
    y_valid_pred = predict(model, X_valid)
    metrics = compute_metrics_for_regression(y_valid, y_valid_pred)

    print(metrics)
    print("final MCRMSE on validation set: ", np.mean(list(metrics.values())))

    # save submission.csv file for the test set
    submission_df = pd.read_csv(
        "test.csv", header=0, names=["text_id", "full_text"], index_col="text_id"
    )
    X_submission = list(submission_df.full_text.to_numpy())
    y_submission = predict(model, X_submission)
    submission_df = pd.DataFrame(y_submission, columns=DIMENSIONS)
    submission_df.index = submission_df.index.rename("text_id")
    submission_df.to_csv("submission.csv")

import pandas as pd


def create_new_dataframe(data, column_names):
    new_data = {}

    for column in column_names:
        if column in data.columns:
            new_data[column] = data[column]
        else:
            new_data[column] = pd.Series(0, index=data.index)

    new_dataframe = pd.DataFrame(new_data)
    return new_dataframe


# Loading the dataset to train a binary classfier downstream
df = pd.read_csv("train.csv")
num_examples = df.shape[0]
df = df.sample(frac=1, random_state=1)
train_data = df[0 : int(0.8 * num_examples)]
val_data = df[int(0.8 * num_examples) + 1 :]

train_data[["Deck", "Cabin_num", "Side"]] = train_data["Cabin"].str.split("/", expand=True)
train_data = train_data.drop("Cabin", axis=1)

val_data[["Deck", "Cabin_num", "Side"]] = val_data["Cabin"].str.split("/", expand=True)
val_data = val_data.drop("Cabin", axis=1)

TargetY = train_data["Transported"]
TargetY_test = val_data["Transported"]

# Expanding features to have boolean values as opposed to categorical
# You can check all the features as column names and try to find good correlations with the target variable
selectColumns = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
ResourceX = pd.get_dummies(train_data[selectColumns])
ResourceX_test = pd.get_dummies(val_data[selectColumns])


# ***********************************************
# In this part of the code, write and train the model on the above dataset to perform the task.
# Note that the output accuracy should be stored in train_accuracy and val_accuracy variables
# ***********************************************


# ***********************************************
# End of the main training module
# ***********************************************

print(f"Train Accuracy: {train_accuracy}")  # noqa: F821
print(f"Validation Accuracy: {val_accuracy}")  # noqa: F821

test_data = pd.read_csv("test.csv")
test_data[["Deck", "Cabin_num", "Side"]] = test_data["Cabin"].str.split("/", expand=True)
test_data = test_data.drop("Cabin", axis=1)

test_X = pd.get_dummies(test_data[selectColumns])
test_X.insert(loc=17, column="Deck_T", value=0)

test_preds = model.predict(test_X)  # noqa: F821


output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Transported": test_preds})
output.to_csv("submission.csv", index=False)

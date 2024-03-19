# Import helpful libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = "train.csv"
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# You can change the features needed for this task depending on your understanding of the features and the final task
features = [
    "MSSubClass",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# ***********************************************
# In this part of the code, write and train the model on the above dataset to perform the task.
# This part should populate the variable train_mae and valid_mae on the model selected
# ***********************************************


# ***********************************************
# End of the main training module
# ***********************************************

print("Train MAE: {:,.0f}".format(train_mae))  # noqa: F821
print("Validation MAE: {:,.0f}".format(valid_mae))  # noqa: F821

test_data = pd.read_csv("test.csv")
test_X = test_data[features]
test_preds = model.predict(test_X)  # noqa: F821

output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_preds})
output.to_csv("submission.csv", index=False)

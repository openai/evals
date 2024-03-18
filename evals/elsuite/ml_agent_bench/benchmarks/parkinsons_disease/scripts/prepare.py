import random
from pathlib import Path
from shutil import copyfile

import pandas as pd

from evals.elsuite.ml_agent_bench.utils import get_root_dir

env_dir = Path(__file__).parent / ".." / "env"
dataset_dir = (
    get_root_dir() / "registry" / "data" / "ml_agent_bench" / "parkinsons_disease" / "dataset"
)


if not dataset_dir.is_dir():
    dataset_dir.mkdir(parents=False, exist_ok=False)

    input(
        "Please download the data at https://www.kaggle.com/"
        f"competitions/amp-parkinsons-disease-progression-prediction/data "
        f"into {dataset_dir}. Press any key after you've downloaded "
        "the data to continue."
    )


# check required files exist

proteins_csv = dataset_dir / "train_proteins.csv"
clinical_csv = dataset_dir / "train_clinical_data.csv"
peptides_csv = dataset_dir / "train_peptides.csv"
supplemental_csv = dataset_dir / "supplemental_clinical_data.csv"
utils_py = dataset_dir / "public_timeseries_testing_util.py"

assert proteins_csv.is_file(), f"{proteins_csv} does not exist!"
assert clinical_csv.is_file(), f"{clinical_csv} does not exist!"
assert peptides_csv.is_file(), f"{peptides_csv} does not exist!"
assert supplemental_csv.is_file(), f"{supplemental_csv} does not exist!"
assert utils_py.is_file(), f"{utils_py} does not exist!"


# create example files directory in env

example_test_files_dir = env_dir / "example_test_files"
example_test_files_dir.mkdir(parents=False, exist_ok=True)


# split train to train and test in env

data_proteins = pd.read_csv(proteins_csv)
data_clinical = pd.read_csv(clinical_csv)
data_peptides = pd.read_csv(peptides_csv)
data_supplemental = pd.read_csv(supplemental_csv)

random.seed(42)

patient_id = data_clinical["patient_id"].unique()
test_patient_id = random.sample(patient_id.tolist(), 2)
train_patient_id = [x for x in patient_id if x not in test_patient_id]

data_proteins[data_proteins["patient_id"].isin(train_patient_id)].to_csv(
    env_dir / "train_proteins.csv", index=False
)
data_clinical[data_clinical["patient_id"].isin(train_patient_id)].to_csv(
    env_dir / "train_clinical_data.csv", index=False
)
data_peptides[data_peptides["patient_id"].isin(train_patient_id)].to_csv(
    env_dir / "train_peptides.csv", index=False
)
data_supplemental[data_supplemental["patient_id"].isin(train_patient_id)].to_csv(
    env_dir / "supplemental_clinical_data.csv", index=False
)
data_proteins[data_proteins["patient_id"].isin(test_patient_id)].to_csv(
    env_dir / "example_test_files" / "test_proteins.csv", index=False
)
data_peptides[data_peptides["patient_id"].isin(test_patient_id)].to_csv(
    env_dir / "example_test_files" / "test_peptides.csv", index=False
)
test_clinical = data_clinical[data_clinical["patient_id"].isin(test_patient_id)]


# copy utils file

copyfile(
    src=utils_py,
    dst=env_dir / utils_py.name,
)

# create example test.csv

temp_list = []
for i in range(1, 5):
    temp = test_clinical.copy()
    temp["level_3"] = i
    temp["updrs_test"] = f"updrs_{i}"
    temp_list.append(temp)
mock_train = pd.concat(temp_list)
mock_train["row_id"] = mock_train[["patient_id", "visit_month", "level_3"]].apply(
    (lambda r: f"{r.patient_id}_{int(r.visit_month)}_updrs_{r.level_3}"), axis=1
)
mock_train[["visit_id", "patient_id", "visit_month", "row_id", "updrs_test"]].to_csv(
    env_dir / "example_test_files" / "test.csv", index=False
)

# Create sample_submission.csv

temp_list = []
for wait in [0, 6, 12, 24]:
    temp = mock_train.copy()
    temp["wait"] = wait
    temp_list.append(temp)
y = pd.concat(temp_list)
y = y[y.visit_month + y.wait <= 108]
y["prediction_id"] = y[["patient_id", "visit_month", "wait", "level_3"]].apply(
    (lambda r: f"{r.patient_id}_{int(r.visit_month)}_updrs_{r.level_3}_plus_{r.wait}_months"),
    axis=1,
)


def get_rating(row):
    rating = test_clinical[
        test_clinical["visit_id"] == f"{row.patient_id}_{int(row.visit_month) + int(row.wait)}"
    ][f"updrs_{row.level_3}"]
    if len(rating) == 0:
        return None
    return rating.item()


y["rating"] = y[["patient_id", "visit_month", "wait", "level_3"]].apply(get_rating, axis=1)
y = y.dropna()
y[["prediction_id", "rating", "visit_month"]].to_csv("answer.csv", index=False)

y["rating"] = 0
y[["prediction_id", "rating", "visit_month"]].to_csv(
    env_dir / "example_test_files" / "sample_submission.csv", index=False
)

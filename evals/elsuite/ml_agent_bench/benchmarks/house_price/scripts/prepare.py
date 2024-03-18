from pathlib import Path

import pandas as pd

from evals.elsuite.ml_agent_bench.utils import get_root_dir

env_dir = Path(__file__).parent / ".." / "env"
script_dir = Path(__file__).parent
dataset_dir = get_root_dir() / "registry" / "data" / "ml_agent_bench" / "house_price" / "dataset"

if not dataset_dir.is_dir():
    dataset_dir.mkdir(parents=False, exist_ok=False)

    input(
        "Please download the data at https://www.kaggle.com/"
        f"competitions/home-data-for-ml-course/data "
        f"into {dataset_dir}. Press any key after you've downloaded "
        "the data to continue."
    )


train = pd.read_csv(dataset_dir / "train.csv")
train = train.reset_index(drop=True)
train.iloc[: int(len(train) * 0.8)].to_csv(env_dir / "train.csv", index=False)
test = train.iloc[int(len(train) * 0.8) :]

test.drop(list(train.keys())[1:-1], axis=1).to_csv(script_dir / "answer.csv", index=False)
test = test.drop(["SalePrice"], axis=1).to_csv(env_dir / "test.csv", index=False)

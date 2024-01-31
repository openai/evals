import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union, List, Any
import traceback

import numpy as np
import pandas as pd
from PIL import Image

try:
    import wandb
except:
    print("No wandb found!")


class WandBReporter:
    @staticmethod
    def _convert_logger_table(df: pd.DataFrame) -> wandb.Table:
        wandb_df = deepcopy(df)
        if wandb_df.shape[0] == 0:
            return wandb.Table(wandb_df)
        for col in wandb_df.columns:
            i = 0
            while not wandb_df[col].iloc[i]:
                i += 1
                if i == wandb_df.shape[0]:
                    i = 0
                    break
            data0 = wandb_df[col].iloc[i]
            # if isinstance(data0, Chem.Mol):
            #     aim_df[col] = [wandb.Molecule(molfile) if molfile else None for molfile in molfiles]
            if isinstance(data0, Image.Image):
                for i, img in enumerate(wandb_df[col]):
                    wandb_df.loc[i, col] = wandb.Image(img) if img else None
        return wandb.Table(dataframe=wandb_df, allow_mixed_types=True)

    @staticmethod
    def _convert_logger_data(v: Any) -> Any:
        import matplotlib.pyplot as plt
        try:
            import plotly.graph_objects as go
        except ImportError:
            go = plt
        if type(v) in [go.Figure, plt.Figure]:
            return wandb.Plotly(v)
        if type(v) in [Image.Image] or (type(v) == str and Path(v).exists() and Path(v).suffix in [".png", ".jpg"]):
            return wandb.Image(v)
        if type(v) in [pd.DataFrame]:
            return WandBReporter._convert_logger_table(v)
        if type(v) in [np.ndarray, list]:
            return wandb.Histogram(v)
        if type(v) == str:
            return v
        return v
    
    @staticmethod
    def report_run(config_logger: Dict, config_run: Dict = {}, logger_data: Dict = {}, step: int = -1):
        wandb_config = config_logger.get("wandb", {}).copy()
        wandb_config["name"] = config_logger["name"]
        wandb_config["group"] = config_logger["group"]
        wandb_config["id"] = config_logger["id"]
        wandb_config["config"] = config_run
        wandb_config["tags"] = list(set([config_logger["name"]] + wandb_config.get("tags", [])))
        wandb.login(key=os.environ.get("WANDB_API_KEY", None))
        if "entity" not in wandb_config:
            wandb_config["entity"] = os.environ.get("WANDB_ENTITY", "uni-finder")
        if "project" not in wandb_config:
            wandb_config["project"] = config_logger.get("project", os.environ.get("WANDB_PROJECT", "LLM-Eval"))

        try:
            run = wandb.init(**wandb_config)
        except:
            traceback.print_exc()
            wandb_config["mode"] = "offline"
            run = wandb.init(**wandb_config)

        logger_data_wandb = {key: WandBReporter._convert_logger_data(value) for key, value in logger_data.items()}

        if step >= 0:
            wandb.log(data=logger_data_wandb, step=step)
        else:
            wandb.log(data=logger_data_wandb)
        wandb.finish()

    @staticmethod
    def report_summary():
        pass

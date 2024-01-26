from pathlib import Path
from typing import Dict, Union, List
import traceback

import pandas as pd

try:
    import wandb
except:
    print("No wandb found!")


class WandBReporter:
    @staticmethod
    def report_run(config_logger: Dict, metric_data: pd.DataFrame, step: int = -1):
        logger_data = {}

        logger_data[f"correlation_ligand_sidechain"] = wandb.Plotly(fig)

        wandb_config = config_logger.get("wandb", {}).copy()
        wandb_config["name"] = config_logger["name"]
        wandb_config["group"] = config_logger["group"]
        wandb_config["id"] = config_logger["id"]
        wandb.login(key=wandb_config.pop('key'))

        try:
            run = wandb.init(**wandb_config)
        except:
            traceback.print_exc()
            wandb_config["mode"] = "offline"
            run = wandb.init(**wandb_config)
        sampler_metric_wb = wandb.Table(dataframe=metric_data)
        logger_data["sampler_metrics"] = sampler_metric_wb

        if step >= 0:
            wandb.log(data=logger_data, step=step)
        else:
            wandb.log(data=logger_data)
        wandb.finish()

    @staticmethod
    def report_summary():
        pass

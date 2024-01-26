import glob
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, List, Any

import numpy as np
import pandas as pd
import aim
from PIL import Image


class DPTrackingReporter:
    @staticmethod
    def _convert_logger_table(df: pd.DataFrame) -> aim.Table:
        aim_df = deepcopy(df)
        if aim_df.shape[0] == 0:
            return aim.Table(aim_df)
        for col in aim_df.columns:
            i = 0
            while not aim_df[col].iloc[i]:
                i += 1
                if i == aim_df.shape[0]:
                    i = 0
                    break
            data0 = aim_df[col].iloc[i]
            # if isinstance(data0, Chem.Mol):
            #     molfiles = []
            #     tmpdir = f"aim-tmp-{uuid.uuid4().hex}"
            #     Path(tmpdir).mkdir(exist_ok=True, parents=True)
            #     for i, mol in enumerate(aim_df[col]):
            #         if mol:
            #             molfile = f"{tmpdir}/{i}.sdf"
            #             Chem.MolToMolFile(mol, molfile)
            #             molfiles.append(molfile)
            #         else:
            #             molfiles.append(None)
            #     aim_df[col] = [aim.Molecule(molfile) if molfile else None for molfile in molfiles]
            if isinstance(data0, Image.Image):
                imgfiles = []
                tmpdir = f"aim-tmp-{uuid.uuid4().hex}"
                Path(tmpdir).mkdir(exist_ok=True, parents=True)
                for i, img in enumerate(aim_df[col]):
                    if img:
                        imgfile = f"{tmpdir}/{i}.png"
                        img.save(imgfile)
                        imgfiles.append(imgfile)
                    else:
                        imgfiles.append(None)
                aim_df[col] = [aim.TableImage(imgfile) if imgfile else None for imgfile in imgfiles]
        return aim.Table(aim_df)

    @staticmethod
    def _convert_logger_data(v: Any) -> Any:
        import matplotlib.pyplot as plt
        try:
            import plotly.graph_objects as go
        except ImportError:
            go = plt
        if type(v) in [go.Figure, plt.Figure]:
            return aim.Figure(v)
        if type(v) in [Image.Image] or (type(v) == str and Path(v).exists() and Path(v).suffix in [".png", ".jpg"]):
            return aim.Image(v)
        if type(v) in [pd.DataFrame]:
            return DPTrackingReporter._convert_logger_table(v)
        if type(v) in [np.ndarray, list]:
            return aim.Distribution(v)
        if type(v) == str:
            return aim.Text(v)
        return v

    @staticmethod
    def report_run(config_logger: Dict, config_run: Dict = {}, logger_data: Dict = {}, step: int = -1):
        dp_mlops_config = config_logger["dp_mlops"]

        # Experiment Tracking
        if "aim_personal_token" in dp_mlops_config.keys():
            os.environ["AIM_ACCESS_TOKEN"] = dp_mlops_config["aim_personal_token"]
        run = aim.Run(
            experiment=config_logger["project"],
            run_hash=config_logger.get("hash", None),
            repo=dp_mlops_config["aim_repo"]
        )
        run.name = config_logger["name"]
        run["config"] = config_run
        for tag in set([config_logger["name"]] + dp_mlops_config.get("tags", [])):
            if tag and tag.lower() not in [t.lower() for t in run.props.tags]:
                print(tag.lower(), run.props.tags)
                run.add_tag(tag.lower())

        logger_data_aim = {key: DPTrackingReporter._convert_logger_data(value) for key, value in logger_data.items()}

        for key, value in logger_data_aim.items():
            print(key, type(value))
            if "/" not in key or "kcal/mol" in key or "10.1021/" in key or "10.1016/" in key:
                run.track(value, name=key)
            else:
                key, context_str = key.split("/")
                context_dict = {k: v for k, v in [kv.split(":") for kv in context_str.split(",")]}
                run.track(value, name=key, context={**context_dict})
        run.close()

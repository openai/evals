from typing import Optional, Tuple, Union, List
import json
import random

import networkx as nx
import numpy as np
import pandas as pd

from evals.elsuite.identifying_variables.structs import Sample
from evals.elsuite.identifying_variables.renderers.base import RendererBase
from evals.elsuite.identifying_variables.latent_funcs import (
    DISTRIBUTIONS,
    LATENT_FUNC_MAP,
)
from evals.elsuite.identifying_variables.constants import NUM_OBS


def apply_noise(
    data_df: pd.DataFrame, np_rng: np.random.Generator, snr: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply noise to a pandas DataFrame to achieve a specified Signal-to-Noise Ratio
    (SNR).

    Args:
    data_df (pd.DataFrame): The DataFrame containing the original data.
    snr (float): The desired Signal-to-Noise Ratio in decibels (dB).
        If None, no noise is applied.
    """
    if snr is None:
        return data_df

    desired_snr_linear = 10 ** (snr / 10)

    signal_powers = data_df.var()
    noise_powers = signal_powers / desired_snr_linear

    noise = pd.DataFrame(
        np_rng.normal(0, np.sqrt(noise_powers), data_df.shape),
        columns=data_df.columns,
    )
    noisy_df = data_df + noise

    return noisy_df


def sparsify_data(
    data_df: pd.DataFrame, variable_metadata: dict, np_rng: np.random.Generator
) -> pd.DataFrame:
    total_obs = data_df.shape[0]
    for var in variable_metadata.keys():
        sparsity_rate = variable_metadata[var]["extra"]["sparsity_rate"]
        num_missing_obs = int(sparsity_rate * total_obs)
        missing_obs_indices = np_rng.choice(total_obs, num_missing_obs, replace=False)
        data_df.loc[missing_obs_indices, var] = np.nan
    return data_df


class TabularRenderer(RendererBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_obs = NUM_OBS

    def _render_table(self, sample: Sample) -> pd.DataFrame:
        variable_metadata = sample.variable_metadata
        sample_metadata = sample.sample_metadata
        n_obs_samples = self.num_obs
        causal_graph = sample.causal_graph

        # "topological sort" from least to most ancestors (i.e. least to most dependent)
        sorted_vars = nx.topological_sort(causal_graph)
        # necessary so that we can generate data in the correct order

        data_dict = {}
        for var in sorted_vars:
            gen_method = variable_metadata[var]["gen_method"]["name"]
            if "input_x" not in variable_metadata[var]["gen_method"]:
                distr = DISTRIBUTIONS[gen_method]
                distr_kwargs = variable_metadata[var]["gen_method"]["kwargs"]
                data_dict[var] = distr(
                    num_samples=n_obs_samples, **distr_kwargs, rng=self.np_rng
                )
            else:
                latent_func = LATENT_FUNC_MAP[gen_method]
                latent_func_kwargs = variable_metadata[var]["gen_method"]["kwargs"]
                input_x = variable_metadata[var]["gen_method"]["input_x"]
                data_dict[var] = latent_func(x=data_dict[input_x], **latent_func_kwargs)

        data_df = pd.DataFrame(data_dict)

        # apply noise after generating data
        data_df = apply_noise(data_df, self.np_rng, sample_metadata["snr"])
        # apply sparsification after generating and noise
        data_df = sparsify_data(data_df, variable_metadata, self.np_rng)

        # round to 3 decimal places
        data_df = data_df.round(3)

        return data_df


class MarkdownTableRenderer(TabularRenderer):
    """
    Renders tabular data as a markdown table with variable names as column names.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def render_obs(self, sample: Sample) -> str:
        data_df = self._render_table(sample)
        return data_df.to_markdown(index=False)


class CSVTableRenderer(TabularRenderer):
    """
    Renders tabular data as a comma-separated-values (CSV) file with variable names as
    column names.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def render_obs(self, sample: Sample) -> str:
        data_df = self._render_table(sample)
        return data_df.to_csv(index=False)


class JSONTableRenderer(TabularRenderer):
    """
    Renders tabular data as a JSON object with variable names as keys and lists of
    values as values.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def render_obs(self, sample: Sample) -> str:
        data_df = self._render_table(sample)
        return json.dumps(data_df.to_dict(orient="list"))


class LanguageTableRenderer(TabularRenderer):
    """
    Renders tabular data as a natural language description of the data.
    Describing the data row by row.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_obs = 10  # set it to 10
        # realistically no one would read more than 10 rows of data one by one

    def render_obs(self, sample: Sample) -> str:
        data_df = self._render_table(sample)
        variables = list(data_df.columns)
        rendered_obs = ""
        current_step = "first"
        for row in data_df.itertuples(index=False, name=None):
            rendered_obs += self._render_row(row, variables, current_step) + "\n"
            current_step = "next"
        return rendered_obs

    def _render_row(
        self, row: Tuple[Union[int, float]], variables: List[str], current_step: str
    ) -> str:
        string = f"On the {current_step} step, "
        past_participle_verb = self.rng.choice(["measured", "recorded", "reported"])
        for value, var in zip(row, variables):
            if np.isnan(value):
                string += f"{var} was not {past_participle_verb}. "
            else:
                string += (
                    f"{var} was {past_participle_verb} to be {format_number(value)}. "
                )
        return string


def format_number(number: Union[int, float]):
    """Get's rid of trailing .0's"""
    if float(number).is_integer():
        return int(number)
    else:
        return number


if __name__ == "__main__":
    # just for quick testing
    np_rng = np.random.default_rng(0)
    renderer = LanguageTableRenderer(random.Random(0), np_rng)

    from evals.elsuite.identifying_variables.scripts.gen_data import gen_samples

    samples = gen_samples(10, None, np_rng)

    for sample in samples:
        print(nx.to_dict_of_lists(sample.causal_graph))
        print(sample.variable_metadata)
        print(renderer.render_obs(sample))
        print("================")

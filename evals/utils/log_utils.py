import json
from pathlib import Path
from typing import Union


def get_final_results_from_dir(log_dir: Union[str, Path]) -> dict[Path, dict]:
    """
    Given a directory of log files, return a dictionary mapping log file paths to final results.
    """
    final_results_dict = {}
    for path in Path(log_dir).glob("**/*.log"):
        final_results = extract_final_results(path)
        final_results_dict[path] = final_results
    return final_results_dict


def get_specs_from_dir(log_dir: Union[str, Path]) -> dict[Path, dict]:
    """
    Given a directory of log files, return a dictionary mapping log file paths to specs.
    """
    specs_dict = {}
    for path in Path(log_dir).glob("**/*.log"):
        spec = extract_spec(path)
        specs_dict[path] = spec
    return specs_dict


def extract_final_results(path: Path) -> dict:
    """
    Given a path to a log file, find and return the "final_report" dictionary.
    """
    with path.open() as f:
        for line in f.readlines():
            line = line.strip()
            try:
                loaded_line = json.loads(line)
                if "final_report" in loaded_line:
                    return loaded_line["final_report"]
            except json.decoder.JSONDecodeError:
                print(f"Skipping line: {line}")
                continue
    raise ValueError(f"Could not find final_report in {path}")


def extract_individual_results(path: Path, type_string: str = "metrics") -> list[dict]:
    """
    Given a path to a log file, grab all the individual sample results.
    """
    all_data = []
    with path.open() as f:
        for line in f.readlines():
            line = line.strip()
            try:
                loaded_line = json.loads(line)
                if "type" in loaded_line:
                    if loaded_line["type"] == type_string:
                        all_data.append(loaded_line["data"])
            except json.decoder.JSONDecodeError:
                print(f"Skipping line: {line}")
                continue
    return all_data


def extract_spec(path: Path) -> dict:
    """
    Given a path to a log file, find and return the "spec" dictionary.
    """
    with path.open() as f:
        for line in f.readlines():
            line = line.strip()
            try:
                loaded_line = json.loads(line)
                if "spec" in loaded_line:
                    return loaded_line["spec"]
            except json.decoder.JSONDecodeError:
                print(f"Skipping line: {line}")
                continue
    raise ValueError(f"Could not find spec in {path}")

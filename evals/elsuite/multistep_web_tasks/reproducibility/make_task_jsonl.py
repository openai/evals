import json
from pathlib import Path


def main():
    reproducibility_dir = Path(__file__).parents[0].resolve()
    parent_dir = reproducibility_dir.parents[1].resolve()
    data_dir = parent_dir / "evals/registry/data/multistep-web-tasks"
    raw_json = reproducibility_dir / "all_tasks.json"
    with raw_json.open("r") as f:
        all_tasks = json.load(f)

    write_jsonl(data_dir / "all_tasks.jsonl", all_tasks)

    easy_tasks = build_easy_tasks(all_tasks)
    write_jsonl(data_dir / "easy_tasks.jsonl", easy_tasks)

    medium_tasks = build_medium_tasks(all_tasks)
    write_jsonl(data_dir / "medium_tasks.jsonl", medium_tasks)

    hard_tasks = build_hard_tasks(all_tasks)
    write_jsonl(data_dir / "hard_tasks.jsonl", hard_tasks)

    build_and_write_individual_tasks(all_tasks, data_dir)


def select_tasks_by_id(all_tasks: list[dict], task_ids: list[int]):
    return [task for task in all_tasks if task["task_id"] in task_ids]


def build_and_write_individual_tasks(all_tasks: list[dict], data_dir: Path) -> None:
    for i in range(1, 10):
        task: list[dict] = select_tasks_by_id(all_tasks, [i])
        write_jsonl(data_dir / f"task_{i}.jsonl", task)


def build_easy_tasks(all_tasks: list[dict]) -> list[dict]:
    task_ids = [1, 2, 3]
    return select_tasks_by_id(all_tasks, task_ids)


def build_medium_tasks(all_tasks: list[dict]) -> list[dict]:
    task_ids = [4, 5, 6]
    return select_tasks_by_id(all_tasks, task_ids)


def build_hard_tasks(all_tasks: list[dict]) -> list[dict]:
    task_ids = [7, 8, 9]
    return select_tasks_by_id(all_tasks, task_ids)


def write_jsonl(outfile: Path, json_objects: list[dict]) -> None:
    with outfile.open("w") as f:
        for obj in json_objects:
            f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()

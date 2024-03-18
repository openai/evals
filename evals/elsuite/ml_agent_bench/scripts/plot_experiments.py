# %%

import os
import json
import textwrap

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evals.elsuite.ml_agent_bench.utils import get_root_dir

# %%

commit_hash = os.popen("git rev-parse HEAD").read().strip()

commits_to_include = [commit_hash]
run_ids_to_exclude = []
tasks_to_exclude = [
    # v1
    # "ml-agent-bench.vectorization",
    # "ml-agent-bench.parkinsons-disease",
    # "ml-agent-bench.spaceship-titanic",
    # "ml-agent-bench.cifar10",
    # "ml-agent-bench.imdb",
    # "ml-agent-bench.feedback",
    # "ml-agent-bench.ogbn-arxiv",
    # "ml-agent-bench.house-price",
    # v2
    # "ml-agent-bench.ant",
    # "ml-agent-bench.bipedal-walker",
    # "ml-agent-bench.cartpole",
    # "ml-agent-bench.humanoid",
    # "ml-agent-bench.inverted-pendulum",
    # "ml-agent-bench.pong",
    # "ml-agent-bench.pusher",
]

log_files = []

for commit in commits_to_include:
    log_dir = get_root_dir() / "elsuite" / "ml_agent_bench" / "scripts" / "logs" / commit
    log_files += [f for f in log_dir.glob("**/*.log")]

final_reports = []

for log_file in log_files:
    with open(log_file, "r") as f:
        lines = f.readlines()

    completion_fn = None
    eval_name = None

    for line in lines:
        content = json.loads(line)

        if "spec" not in content:
            continue

        if "completion_fns" not in content["spec"]:
            continue

        if "eval_name" not in content["spec"]:
            continue

        assert len(content["spec"]["completion_fns"]) == 1

        completion_fn = content["spec"]["completion_fns"][0]
        eval_name = content["spec"]["eval_name"]
        run_id = content["spec"]["run_id"]

    if completion_fn is None:
        continue

    if eval_name is None:
        continue

    if eval_name in tasks_to_exclude:
        continue

    if run_id is None:
        continue

    if run_id in run_ids_to_exclude:
        continue

    final_report = None

    for line in lines:
        content = json.loads(line)

        if "final_report" not in content:
            continue

        final_report = content["final_report"]

        assert "model_score_humanrelative" in final_report
        assert "model_score" in final_report
        assert "naive_baseline_score" in final_report
        assert "human_baseline_score" in final_report

    if final_report is None:
        continue

    final_reports.append(
        {
            "solver_id": completion_fn,
            "task_id": eval_name,
            "score": final_report["model_score_humanrelative"],
        }
    )

    final_reports.append(
        {
            "solver_id": f"{completion_fn} (raw)",
            "task_id": eval_name,
            "score": final_report["model_score"],
        }
    )

    final_reports.append(
        {
            "solver_id": "naive (raw)",
            "task_id": eval_name,
            "score": final_report["naive_baseline_score"],
        }
    )

    final_reports.append(
        {
            "solver_id": "human (raw)",
            "task_id": eval_name,
            "score": final_report["human_baseline_score"],
        }
    )


# %%

df = pd.DataFrame.from_records(final_reports)
df

# %%

filtered_df = df[~df["solver_id"].str.contains("raw")]
grouped = filtered_df.groupby(["solver_id"])
score_mean = grouped["score"].mean().rename("score")
score_sem = grouped["score"].sem().rename("sem")
report_task_table = pd.concat([score_mean, score_sem], axis=1).reset_index()

report_task_table

# %%

filtered_df = df[df["solver_id"].str.contains("raw")]
grouped = filtered_df.groupby(["solver_id", "task_id"])
score_mean = grouped["score"].mean().rename("score")
score_sem = grouped["score"].sem().rename("sem")
report_summary_table = pd.concat([score_mean, score_sem], axis=1).reset_index()

report_summary_table

# %%

df_non_raw = df[~df["solver_id"].str.contains("raw")]  # drop raw scores

# %%

model_mapping = {
    "human": "Human",
    "naive": "Naive Baseline",
    "ml_agent_bench/baseline/gpt-3.5-turbo-16k": "GPT-3.5 (huang-inspired)",
    "ml_agent_bench/baseline/gpt-4-1106-preview": "GPT-4 (huang-inspired)",
    "generation/direct/gpt-3.5-turbo-16k": "GPT-3.5 (direct)",
    "generation/direct/gpt-4-1106-preview": "GPT-4 (direct)",
    "generation/direct/gemini-pro": "Gemini Pro",
    "generation/direct/llama-2-13b-chat": "LLaMA-2 Chat (13B)",
    "generation/direct/llama-2-70b-chat": "LLaMA-2 Chat (70B)",
    "generation/direct/mixtral-8x7b-instruct": "Mixtral-8x7B Instruct",
}

task_mapping = {
    "ml-agent-bench.babylm.v0": "BabyLM",
    "ml-agent-bench.cifar10.v0": "CIFAR-10",
    "ml-agent-bench.clrs.v0": "CLRS",
    "ml-agent-bench.fathomnet.v0": "FathomNet",
    "ml-agent-bench.feedback.v0": "Feedback",
    "ml-agent-bench.house-price.v0": "House Prices",
    "ml-agent-bench.identify-contrails.v0": "Identify Contrails",
    "ml-agent-bench.imdb.v0": "IMDb",
    "ml-agent-bench.parkinsons-disease.v0": "Parkinson's Disease",
    "ml-agent-bench.llama-inference.v0": "Llama Inference",
    "ml-agent-bench.ogbn-arxiv.v0": "OGBN-ArXiv",
    "ml-agent-bench.spaceship-titanic.v0": "Spaceship Titanic",
    "ml-agent-bench.vectorization.v0": "Vectorization",
    "ml-agent-bench.ant.gpu.v0": "Ant",
    "ml-agent-bench.bipedal-walker.v0": "Bipedal Walker",
    "ml-agent-bench.cartpole.v0": "Cart Pole",
    "ml-agent-bench.humanoid.gpu.v0": "Humanoid",
    "ml-agent-bench.inverted-pendulum.v0": "Inverted Pendulum",
    "ml-agent-bench.pong.gpu.v0": "Pong",
    "ml-agent-bench.pusher.v0": "Pusher",
}

df_non_raw["solver"] = df_non_raw["solver_id"].map(model_mapping)
df_non_raw["task"] = df_non_raw["task_id"].map(task_mapping)

df_non_raw

# %%

task_categories = {
    "Canonical Tasks": [
        "CIFAR-10",
        "IMDb",
        "OGBN-ArXiv",
    ],
    "Kaggle (Classic)": [
        "House Prices",
        "Spaceship Titanic",
    ],
    "Kaggle (Modern)": [
        "Feedback",
        "Parkinson's Disease",
    ],
    "Improve Code": [
        "Llama Inference",
        "Vectorization",
    ],
    "Reinforcement Learning": [
        "Ant",
        "Bipedal Walker",
        "Cart Pole",
        "Humanoid",
        "Inverted Pendulum",
        "Pong",
        "Pusher",
    ],
}

task_to_category = {task: category for category, tasks in task_categories.items() for task in tasks}

task_to_category

# %%

category_colors = {
    "Canonical Tasks": "skyblue",
    "Kaggle (Classic)": "lightgreen",
    "Kaggle (Modern)": "lightcoral",
    "Improve Code": "lightgoldenrodyellow",
    "Reinforcement Learning": "violet",
}

# %%

df_only_direct = df_non_raw[df_non_raw["solver_id"].str.contains("direct|human|naive", regex=True)]
df_only_direct

# %%

rl_report_summary_table = report_summary_table.copy()

rl_report_summary_table["task"] = rl_report_summary_table["task_id"].map(task_mapping)
rl_report_summary_table["category"] = rl_report_summary_table["task"].map(task_to_category)

rl_report_summary_table = rl_report_summary_table[
    rl_report_summary_table["category"] == "Reinforcement Learning"
]

rl_report_summary_table = rl_report_summary_table.sort_values(by=["category", "task", "solver_id"])

rl_report_summary_table

# %%

grouped = df_non_raw.groupby(["task", "solver"])
score_mean = grouped["score"].mean().rename("score")
score_sem = grouped["score"].sem().rename("sem")
plot_df = pd.concat([score_mean, score_sem], axis=1).reset_index()

plot_df

# %%

plot_df["category"] = plot_df["task"].map(task_to_category)
plot_df = plot_df.sort_values(by=["category", "task", "solver"])

plot_df

# %%

palette = {
    # OpenAI
    "GPT-3.5 (huang-inspired)": "#0055ff",
    "GPT-3.5 (direct)": "#78a5ff",
    "GPT-4 (huang-inspired)": "#fc5e03",
    "GPT-4 (direct)": "#ff9c63",
    # Google
    "Gemini Pro": "#ff00ff",
    # Meta
    "LLaMA-2 Chat (13B)": "#ff0000",
    "LLaMA-2 Chat (70B)": "#ff7f7f",
    # Mistral AI
    "Mixtral-8x7B Instruct": "#00ff00",
    # Baselines
    "Human": "#00a318",
    "Naive Baseline": "#c90022",
}

plt.figure(figsize=(10, 8))

ax = sns.barplot(
    data=plot_df,
    x="task",
    y="score",
    hue="solver",
    errorbar=None,
    palette=palette,
    zorder=3,
)

num_hue_levels = len(plot_df["solver"].unique())
bar_group_width = ax.patches[0].get_width() * num_hue_levels

for i, task in enumerate(plot_df["task"].unique()):
    task_data = plot_df[plot_df["task"] == task]

    positions = np.linspace(
        start=i - bar_group_width / 2 + bar_group_width / (2 * num_hue_levels),
        stop=i + bar_group_width / 2 - bar_group_width / (2 * num_hue_levels),
        num=num_hue_levels,
    )

    plt.errorbar(
        x=positions,
        y=task_data["score"],
        yerr=task_data["sem"],
        fmt="none",  # This removes the line connecting the error bars
        capsize=5,  # Sets the width of the error bar caps
        color="black",  # Error bar color
        zorder=3,  # Ensure error bars are above the bars but below the legend
        linewidth=1.5,  # Width of the error bar lines
    )

solvers_legend = ax.legend(title="Solvers", loc="upper left", bbox_to_anchor=(1, 1))

plt.gca().add_artist(solvers_legend)

naive_baseline = plt.axhline(
    y=-0.001,
    color="#c90022",
    linestyle="--",
    linewidth=2,
    zorder=2,
    alpha=0.5,
)

human_baseline = plt.axhline(
    y=1,
    color="#00a318",
    linestyle="--",
    linewidth=2,
    zorder=2,
)

naive_baseline_legend = mlines.Line2D(
    [],
    [],
    color="#c90022",
    linestyle="--",
    label="Naive Solution",
)

human_baseline_legend = mlines.Line2D(
    [],
    [],
    color="#00a318",
    linestyle="--",
    label="Human",
)

ax.legend(
    handles=[
        naive_baseline_legend,
        human_baseline_legend,
    ],
    title="Baselines",
    loc="upper left",
    bbox_to_anchor=(1, 0.2),
)

# Feature flag to toggle background colouring
if True:
    for category in task_categories:
        task_categories[category] = [
            task for task in task_categories[category] if task in plot_df["task"].values
        ]

    task_positions = {task: i for i, task in enumerate(plot_df["task"].unique())}

    for category, color in category_colors.items():
        tasks_in_category = task_categories[category]

        if not tasks_in_category:
            continue

        positions = [task_positions[task] for task in tasks_in_category]
        min_pos, max_pos = min(positions), max(positions)

        ax.axvspan(min_pos - 0.5, max_pos + 0.5, color=color, alpha=0.2)

        width = 13

        if category == "Improve Code":
            width = 10

        wrapped_label = textwrap.fill(category, width=width)

        plt.text(
            x=(min_pos + max_pos) / 2,
            y=ax.get_ylim()[1] * 1.00,
            s=wrapped_label,
            ha="center",
            va="center",
            fontsize=10,
        )

plt.xticks(rotation=90)
plt.yticks([x / 10.0 for x in range(-1, 12, 1)])
plt.xlabel("")
plt.ylabel("Human-relative score")
plt.title("Human-relative score for Model, Human and Naive Baseline")
plt.grid(True, zorder=0)

plt.savefig("bar.png", bbox_inches="tight", pad_inches=1)

plt.show()

# %%

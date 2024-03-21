import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
from pathlib import Path
import matplotlib.colors as mcolors
import argparse
import seaborn as sns

from evals.utils.log_utils import extract_spec, get_final_results_from_dir

WINDOW_SIZES = {
    "FrozenLake-v1 {'map_name': '4x4', 'is_slippery': False}": 20,
    "BanditTwoArmedHighLowFixed-v0 {}": 40,
    "BanditTenArmedRandomFixed-v0 {}": 40,
    "CliffWalking-v0 {}": 20,
    "FrozenLake-v1 {'map_name': '4x4', 'is_slippery': False, 'desc': ['SHFF', 'FFFF', 'FFGH', 'HFHF']}": 20,
    "default": 20,
}

PRETTY_MODEL_NAMES = {
    'generation/direct/gpt-4-turbo-preview': 'GPT-4 Turbo Preview',
    'incontext_rl/random': 'Random Strategy',
    'generation/direct/gpt-3.5-turbo': 'GPT-3.5 Turbo',
    'incontext_rl/qlearning_scratch': 'Q-Learning from scratch',
    'incontext_rl/qlearning_trained': 'Q-Learning trained',
    'generation/direct/gemini-pro': 'Gemini Pro 1.0',
    'generation/direct/mixtral-8x7b-instruct': 'Mixtral 8x7b',
}

PRETTY_ENV_TITLES = {
    "FrozenLake-v1 {'map_name': '4x4', 'is_slippery': False}": 'Frozen Lake (4x4, Non-slippery)',
    "BanditTwoArmedHighLowFixed-v0 {}": "Two-Armed Bandit",
    "BanditTenArmedRandomFixed-v0 {}": "Ten-Armed Bandit",
    "CliffWalking-v0 {}": "Cliff Walking",
    "FrozenLake-v1 {'map_name': '4x4', 'is_slippery': False, 'desc': ['SFFF', 'FHFH', 'FFFH', 'GFFH']}": 'Frozen Lake Custom Map (4x4, Non-slippery)',
}

MODEL_STYLES = {
    'generation/direct/gpt-4-turbo-preview': {'line_style': '-', 'color': 'purple', 'alpha': 0.7},
    'incontext_rl/random': {'line_style': ':', 'color': 'grey', 'alpha': 0.7},
    'generation/direct/gpt-3.5-turbo': {'line_style': '-', 'color': 'green', 'alpha': 0.7},
    'incontext_rl/qlearning_scratch': {'line_style': '--', 'color': 'grey', 'alpha': 0.7},
    'incontext_rl/qlearning_trained': {'line_style': '-', 'color': 'black', 'alpha': 0.7},
    'generation/direct/gemini-pro': {'line_style': '-', 'color': 'blue', 'alpha': 0.7},
    'generation/direct/mixtral-8x7b-instruct': {'line_style': '-', 'color': 'orange', 'alpha': 0.7},
    'default': {'line_style': '-', 'color': 'black', 'alpha': 0.5},
}

def calculate_episode_rewards(row: pd.Series) -> list:
    """
    Calculate the rewards for each episode based on the episode end steps and rewards.
    """
    episode_end_steps = row['episode_end_steps']
    rewards = row['rewards']
    episode_rewards = []
    if not episode_end_steps:  # Handle case where there was only 1 episode
        return [sum(rewards)]
    start_index = 0
    for end_index in episode_end_steps:
        episode_reward = sum(rewards[start_index:end_index])
        episode_rewards.append(episode_reward)
        start_index = end_index
    return episode_rewards

def calculate_rolling_average(episode_rewards: list, window_size: int) -> list:
    """
    Calculate the rolling average of the episode rewards using a specified window size.
    """
    window_size = int(window_size)
    rolling_averages = []
    for i in range(len(episode_rewards)):
        # Calculate the start index for the window; ensure it's not negative
        start_index = max(0, i - window_size + 1)
        # Calculate the running average for the current window
        window_average = np.mean(episode_rewards[start_index:i+1])
        rolling_averages.append(window_average)
    return rolling_averages

def calculate_custom_episode_end_steps_for_cliffwalking(rewards: list, existing_end_steps: list) -> list:
    """
    Calculate episode end steps based on rewards and append to existing end steps.
    An episode also ends when the reward is -100 i.e. when the agent falls off the cliff.

    Args:
    rewards (list): List of rewards for each step in an episode.
    existing_end_steps (list): List of already identified episode end steps.

    Returns:
    list: Updated list of indices representing the end of each episode.
    """
    new_end_steps = [i + 1 for i, reward in enumerate(rewards) if reward == -100]
    # Combine existing and new end steps, remove duplicates, and sort
    combined_end_steps = sorted(set(existing_end_steps + new_end_steps))
    return combined_end_steps

def extract_results(datadir: Path) -> pd.DataFrame:
    """
    Extracts results from the specified directory and returns a DataFrame.

    Args:
    datadir (Path): Path to the directory containing the experiment results.

    Returns:
    pd.DataFrame: DataFrame containing the experiment results.
    """
    print(f"Extracting results from directory: {datadir}")
    df_rows = []
    final_results = get_final_results_from_dir(datadir)
    if not final_results:
        print("No results found in directory.")
        raise ValueError("No results found in directory.")

    for path, results in final_results.items():
        print(f"Processing file: {path}")
        spec = extract_spec(path)
        if not spec:
            raise ValueError(f"No spec found for {path}")
        model = spec.get("completion_fns", [None])[0]
        base_eval = spec.get("base_eval")
        if not model or base_eval is None:
            raise ValueError(f"Missing model or base_eval in spec for {path}")
        
        environments = results.get('environments', [])
        for env in environments:
            metrics = env.get('metrics', {})
            flattened_metrics = {f"{k}": v for k, v in metrics.items()} # Flatten metrics into separate columns
            print(f"Extracted {env['env']} metrics for model: {model}")
            
            # Calculate custom episode end steps for CliffWalking environment
            if env['env'] == "CliffWalking-v0 {}":
                rewards = metrics.get('rewards', [])
                existing_end_steps = metrics.get('episode_end_steps', [])
                episode_end_steps = calculate_custom_episode_end_steps_for_cliffwalking(rewards, existing_end_steps)
                flattened_metrics['episode_end_steps'] = episode_end_steps
            
            df_rows.append({"model": model, "base_eval": base_eval, "environment": env['env'], **flattened_metrics})

    df = pd.DataFrame(df_rows)

    if 'episode_rewards' not in df.columns:
        df['episode_rewards'] = df.apply(calculate_episode_rewards, axis=1)

    # For plots
    df['cumulative_episode_rewards'] = df['episode_rewards'].apply(np.cumsum)
    df['average_episode_reward'] = df['episode_rewards'].apply(np.mean)
    df['window_size'] = df['environment'].map(WINDOW_SIZES).fillna(WINDOW_SIZES.get('default', 20))
    df['rolling_average_rewards'] = df.apply(lambda row: calculate_rolling_average(row['episode_rewards'], row['window_size']), axis=1)

    # We also calculate the rolling average across different window sizes
    df['rolling_average_rewards_5_episodes'] = df.apply(lambda row: calculate_rolling_average(row['episode_rewards'], 5), axis=1)
    df['rolling_average_rewards_10_episodes'] = df.apply(lambda row: calculate_rolling_average(row['episode_rewards'], 10), axis=1)
    df['rolling_average_rewards_20_episodes'] = df.apply(lambda row: calculate_rolling_average(row['episode_rewards'], 20), axis=1)
    df['rolling_average_rewards_50_episodes'] = df.apply(lambda row: calculate_rolling_average(row['episode_rewards'], 50), axis=1)
    
    # We also calculate the average reward for the last 5, 10, 20, and 50 episodes. For older runs, we may not have this information.
    if 'average_reward_last_5_episodes' not in df.columns:
        df['average_reward_last_5_episodes'] = df['episode_rewards'].apply(lambda rewards: np.mean(rewards[-5:]))
    if 'average_reward_last_10_episodes' not in df.columns:
        df['average_reward_last_10_episodes'] = df['episode_rewards'].apply(lambda rewards: np.mean(rewards[-10:]))
    if 'average_reward_last_20_episodes' not in df.columns:
        df['average_reward_last_20_episodes'] = df['episode_rewards'].apply(lambda rewards: np.mean(rewards[-20:]))
    if 'average_reward_last_50_episodes' not in df.columns:
        df['average_reward_last_50_episodes'] = df['episode_rewards'].apply(lambda rewards: np.mean(rewards[-50:]))

    print(f"Extraction complete. {len(df_rows)} rows in DataFrame.")
    return df

def plot_rewards(df, environment, reward_type, out_dir, window_size=None):
    """
    Generalized function to plot episode, cumulative, or running average rewards for different models
    on the same graph for a specific environment. It automatically determines the plot type (line or scatter)
    based on the number of episodes and includes the 95% confidence intervals for line plots.

    Args:
    df (pd.DataFrame): DataFrame containing the experiment results.
    environment (str): Name of the environment to plot.
    reward_type (str): Type of reward to plot. Must be one of 'episode_rewards', 'cumulative_episode_rewards', or 'rolling_average_rewards'.
    out_dir (Path): Path to the directory to save the plots.
    window_size (int): Window size for calculating rolling averages. If None, the window size will be determined based on the environment.
    """
    valid_reward_types = ['episode_rewards', 'cumulative_episode_rewards', 'rolling_average_rewards']
    if reward_type not in valid_reward_types:
        raise ValueError(f"Invalid reward_type. Expected one of {valid_reward_types}, got {reward_type}")

    # Filter the DataFrame for the specific environment
    filtered_df = df[df['environment'] == environment]

    # Explode the specified reward list into separate rows and prepare for plotting
    rewards_df = filtered_df.explode(reward_type).reset_index()  # Each row will be a single episode
    rewards_df['episode'] = rewards_df.groupby(['model', 'index']).cumcount() + 1  # Add episode number as a column
    rewards_df['reward'] = rewards_df[reward_type]  # Rename the column for clarity

    truncate_per_model = True
    if environment == "CliffWalking-v0 {}":
        truncate_per_model = False  # Hacky workaround to make better plots since some models only have 1 episode on CliffWalking

    if truncate_per_model:
        filtered_rewards_df = pd.DataFrame()
        for model, group in rewards_df.groupby('model'):
            # Count the number of runs for each episode number
            episode_counts = group.groupby('episode').size()
            # Check if there are at least 3 runs for any episode number
            if episode_counts.max() >= 3:
                # Find the maximum episode number where at least 3 runs are available
                max_episode_with_at_least_3_runs = episode_counts[episode_counts >= 3].index.max()
                # Filter the group DataFrame to only include data up to this episode number
                model_filtered = group[group['episode'] <= max_episode_with_at_least_3_runs]
            else:
                # If there are fewer than 3 runs for all episodes, include all data for this model
                model_filtered = group
            # Append the filtered data for the current model to the overall filtered DataFrame
            filtered_rewards_df = pd.concat([filtered_rewards_df, model_filtered], ignore_index=True)
        rewards_df = filtered_rewards_df

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Determine the plot type based on the number of episodes
    num_episodes = len(rewards_df['episode'].unique())
    if num_episodes > 1:
        # Iterate over each unique model in the DataFrame
        for model in rewards_df['model'].unique():
            # Filter the DataFrame for the current model
            model_df = rewards_df[rewards_df['model'] == model]
            # Get the custom style for the current model using the helper function
            custom_style = MODEL_STYLES.get(model, MODEL_STYLES['default'])
            pretty_model_name = PRETTY_MODEL_NAMES.get(model, model)
            # Plot the data for the current model on the same axes with custom settings
            lineplot = sns.lineplot(data=model_df, x='episode', y='reward', estimator='mean', errorbar=('ci', 95), 
                                    linestyle=custom_style['line_style'], color=custom_style['color'], 
                                    alpha=custom_style['alpha'], label=pretty_model_name, ax=ax,
                                    err_kws={'alpha': 0.035})
            # Add labels to the final value on the x axis
            for line in lineplot.get_lines():
                x, y = line.get_data()
                if len(x) > 0:  # Check if there is data to plot
                    ax.text(x[-1], y[-1], f"{y[-1]:.2f}", color=line.get_color(), fontsize=9)
    else:
        # For a single episode, use scatter plot, differentiating models by color
        scatterplot = sns.scatterplot(data=rewards_df, x='episode', y='reward', hue='model', ax=ax)
        # Add labels to the final value on the x axis
        for line in scatterplot.collections:
            offsets = line.get_offsets()
            if offsets.size > 0:  # Check if there are points to plot
                last_point = offsets[-1]
                ax.text(last_point[0], last_point[1], f"{last_point[1]:.2f}", fontsize=9)

    pretty_env_title = PRETTY_ENV_TITLES.get(environment, environment)
    plt.title(f'{reward_type.replace("_", " ").title()} in {pretty_env_title} (Window Size: {window_size})' if reward_type == 'rolling_average_rewards' else f'{reward_type.replace("_", " ").title()} in {pretty_env_title}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(1, num_episodes)
    plt.tight_layout()
    plot_dir = out_dir / reward_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{environment}.png')
    plt.show()

def calculate_rolling_averages(df: pd.DataFrame, max_items: int = 200):
    """
    Calculate the averaged final and max rolling averages for the first N items in each model and environment.
    Args:
    df (pd.DataFrame): DataFrame containing the experiment results.
    max_items (int): Maximum number of items to consider for calculating rolling averages.
    Returns:
    dict: Dictionary containing the averaged final and max rolling averages for each model and environment.
    """

    model_env_averages_info = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_env_averages_info[model] = {}
        all_final_rolling_averages = []  # To store all final rolling averages across environments for each model
        for env in model_df['environment'].unique():
            env_df = model_df[model_df['environment'] == env]
            # Determine the last shared episode across all runs for the current model and environment, limited to the first max_items items
            max_shared_episode = min(max_items, env_df['rolling_average_rewards'].apply(lambda rewards: len(rewards[:max_items])).min())
            # Truncate each run's rolling_average_rewards to the max shared episode and then calculate averages
            truncated_averages = env_df['rolling_average_rewards'].apply(lambda rewards: rewards[:max_shared_episode])
            final_rolling_averages = round(truncated_averages.apply(lambda rewards: rewards[-1] if len(rewards) > 0 else None).mean(), 2)
            max_rolling_averages = round(truncated_averages.apply(lambda rewards: max(rewards) if len(rewards) > 0 else None).mean(), 2)
            
            all_final_rolling_averages.append(final_rolling_averages)  # Append the final rolling average for the current environment
            
            model_env_averages_info[model][env] = {
                'average_final_rolling_averages': final_rolling_averages,
                'average_max_rolling_averages': max_rolling_averages,
            }
        
        # Calculate the average final rolling average across all environments for the current model
        average_final_across_envs = round(sum(all_final_rolling_averages) / len(all_final_rolling_averages), 2) if all_final_rolling_averages else None
        model_env_averages_info[model]['average_final_rolling_averages_across_envs'] = average_final_across_envs
    return model_env_averages_info

def json_of_results(df: pd.DataFrame, out_dir: Path):
    """
    JSON dump of the results.

    Each model will have the following information, grouping by environment:
    - Average episode reward
    - Last rolling average reward for each of 5, 10, 20, and 50 episodes
    - Max rolling average reward across the 5, 10, 20, and 50 episodes
    - Invalid response rate

    Where there are multiple runs for a model and environment, the average of the above values will be calculated.
    """

    model_info = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_info[model] = {}
        for env in model_df['environment'].unique():
            env_df = model_df[model_df['environment'] == env]
            # Calculate the average rolling averages across all runs for each window size, then find the max
            average_rolling_averages_5 = env_df['rolling_average_rewards_5_episodes'].apply(pd.Series).mean().max()
            average_rolling_averages_10 = env_df['rolling_average_rewards_10_episodes'].apply(pd.Series).mean().max()
            average_rolling_averages_20 = env_df['rolling_average_rewards_20_episodes'].apply(pd.Series).mean().max()
            average_rolling_averages_50 = env_df['rolling_average_rewards_50_episodes'].apply(pd.Series).mean().max()

            model_info[model][env] = {
                'average_episode_reward': round(env_df['average_episode_reward'].mean(), 2),
                'average_reward_last_5_episodes': round(env_df['average_reward_last_5_episodes'].mean(), 2),
                'average_reward_last_10_episodes': round(env_df['average_reward_last_10_episodes'].mean(), 2),
                'average_reward_last_20_episodes': round(env_df['average_reward_last_20_episodes'].mean(), 2),
                'average_reward_last_50_episodes': round(env_df['average_reward_last_50_episodes'].mean(), 2),
                'max_rolling_average_rewards_5_episodes': round(average_rolling_averages_5, 2),
                'max_rolling_average_rewards_10_episodes': round(average_rolling_averages_10, 2),
                'max_rolling_average_rewards_20_episodes': round(average_rolling_averages_20, 2),
                'max_rolling_average_rewards_50_episodes': round(average_rolling_averages_50, 2),
                'invalid_response_rate': round(env_df['invalid_response_rate'].mean(), 2),
            }
    with open(out_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)

def main(log_dir: str = None, out_dir: str = None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=not log_dir)
    parser.add_argument("--out_dir", "-o", type=str, required=not out_dir)
    args = parser.parse_args()
    log_dir = Path(log_dir) if log_dir else Path(args.log_dir)
    out_dir = Path(out_dir) if out_dir else Path(args.out_dir)

    # Extract results from directory
    df = extract_results(log_dir)

    # # Plot episode rewards with 95% confidence intervals
    for env in df['environment'].unique():
        plot_rewards(df, env, 'episode_rewards', out_dir)
        plot_rewards(df, env, 'cumulative_episode_rewards', out_dir)
        window_size = df[df['environment'] == env]['window_size'].iloc[0]
        plot_rewards(df, env, 'rolling_average_rewards', out_dir, window_size)

    # JSON dump of the results
    json_of_results(df, out_dir)
    

if __name__ == "__main__":
    main()


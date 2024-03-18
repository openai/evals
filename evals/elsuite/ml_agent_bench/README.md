# Human-Relative MLAgentBench Eval

This eval measures a model's ability to solve diverse machine learning research tasks. The best-known human performance has been collated for each task, which is used to calculate a “human-relative” percentage for each task attempt; 0% is a naive baseline (e.g. “random guess”), 100% is obtaining the same performance-gain as the best-known human, and 200% is obtaining 2x the performance-gain of said human. Our thanks go to the authors of [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) on which this work was built.

This eval contains the following 15 tasks:

| Task | Description |
| --- | --- |
| Ant | Coordinate the four legs of an ant-like robot to move forward while applying as little torque on each of the eight joints as possible. |
| Bipedal Walker | Make a robot walk to the rightmost end of the screen without falling over. Applying motor torque costs a small amount of points, more optimal agent will get better score. |
| Cart Pole | Prevent a pole attached to a cart from falling over by pushing the cart either left or right at each timestep. |
| CIFAR-10 | Improve model performance as much as possible within 10 training epochs and save per-class probabilities for the test set. |
| Feedback Prize | Train a language model to grade essays written by 8th-12th grade English Language Learners and submit predictions for the test set. |
| House Prices | Train a model to predict the sale price of a house, iterating over different models or feature selections to enhance performance. |
| Humanoid | Make a humanoid robot walk forward as fast as possible without falling over. |
| IMDb | Fine-tune DistilBERT on the IMDb dataset to classify movie reviews and save per-class probabilities for the test set. |
| Inverted Pendulum | Similarly to Cart Pole, the goal is to prevent a pole attached to a cart from falling over by pushing the cart either left or right at each timestep. The cart is simulated in Mujoco physics simulator, allowing for more complex dynamics (such as varying the effects of gravity). |
| OGBN arXiv | Improve model performance within 10 training epochs on the ogbn-arxiv dataset. |
| Parkinson’s Disease | Train a model on Parkinson's Disease data, focusing on improved performance and lower SMAPE scores, then submit the best predictions. |
| Pong | Play first-to-21 Pong where the goal is to deflect the ball into your opponent’s goal. |
| Pusher | Move a cylinder to a target position using a robot arm consisting of a shoulder, elbow, forearm and wrist joints. |
| Spaceship Titanic | Train a model on the Spaceship Titanic dataset, iterating for better performance, and submit the best predictions. |
| Vectorization | ​​Improve the execution speed of a script by vectorizing computations using numpy, focusing on a specified portion of code. |

## Setup

> **⚠️ Warning:** *This eval allows language models to run arbitrary code on your machine. Please ensure that you only run these experiments in a properly sandboxed environment.*

> **ℹ️** *Multiple tasks require a GPU. We comfortably ran our experiments on a [NC64as T4 v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) machine from Microsoft Azure with an attached 2TB SSD.*

The list of dependencies needed to run this eval are found in `requirements.txt`, which can be installed by running:

```bash
pip install -r requirements.txt
```

Some tasks (optionally) require additional dependencies, which can be found in `benchmarks/<taskid>/scripts/requirements.txt` and likewise can be installed by running:

```bash
pip install -r benchmarks/<taskid>/scripts/requirements.txt
```

where `<taskid>` is the name of the task you wish to run (e.g. `ant`).

## Usage

Run with:

```bash
EVALS_SEQUENTIAL=1 oaieval <solver> ml-agent-bench.<taskid>
# This eval doesn't currently support multi-threading.
```

Where `<taskid>` is one of the 8 tasks listed in the Tasks section below (e.g. `ml-agent-bench.vectorization`).

We suggest using `gpt-4-1106-preview` or `ml_agent_bench/baseline/gpt-4-1106-preview` as default choices for `<solver>`.

For example, to evaluate the `ml_agent_bench/baseline/gpt-4-1106-preview` solver on the `CIFAR-10` task, run:

```bash
EVALS_SEQUENTIAL=1 oaieval ml_agent_bench/baseline/gpt-4-1106-preview ml-agent-bench.cifar10
```

See `evals/registry/solvers/ml-agent-bench.yaml` for a full list of recommended Solvers.

## Evaluation Process

Overall, each run is structured as follows:

1. Given the task description, the agent begins the task.
2. At each step, the agent selects an action from the list of tools (e.g. read a file, edit a file, execute script) or gives a final answer.
3. For a selected agent action:
    - If the action is valid, the eval executes the chosen action in the environment.
        - If the agent attempts to use the Stable Baselines module, the environment refuses:
        ```
        The action has been terminated because it used the Stable Baselines module, which is not allowed. Please write your implementation from scratch.
        ```
        - If the action exceeds the action time limit (e.g. by executing a resource-intensive Python script), the environment throws an error:
        ```
        Time's up! The action exceeded the maximum time limit and terminated early.
        ```
    - If the agent fails to choose a valid action, the environment asks for a new response:
    ```
    No valid action found! You must specify an action at each step in the following format:

    Action: the action to take, should be one of the names of the tools
    Action Input: the input to the action as a valid JSON string
    ```
4. The observation from the environment (e.g. the output of the script execution if the agent chose to execute a script) is returned to the agent, and the agent is prompted to select a further action.
5. Steps 2-4 repeat until the agent calls the “final_answer” tool, or until the maximum number of steps is reached (default max steps is 30).
6. When the attempt has ended, an evaluation script is run to assess how well the agent did.

All attempts are subject to a step, overall time and per-action time limit; the exact limits are task-specific, depending on how challening the task is. This prevents the model from indefinitely refining its solution, or from running scripts which take too long to execute.

## Task State

| Key  | Value |
| --- | --- |
| `task_description` | You have access to the following tools:<br>`{tools_prompt}`<br><br>Research Problem: `{research_problem}`<br><br>Always respond in this format exactly:<br><br>Action: the action to take, should be one of the names of the tools<br>Action Input: the input to the action as a valid JSON string |
| `current_state`  | TaskStateMetadata object that tracks various metadata. |

## Metrics

The below are the key metrics of this eval:

| Metric | Interpretation |
| --- | --- |
| `task_name` | Task name |
| `model_score` | Raw score of the evaluated model on the task |
| `naive_baseline_score` | Raw score of a naive baseline e.g. ‘do nothing’ ‘random guess’ etc. |
| `human_baseline_score` | Raw score of the human baseline, the best-known human performance on this task |
| `model_score_normalized` | Evaluated model score normalised between 0 and 1 |
| `naive_baseline_score_normalized` | Naive baseline score normalised between 0 and 1 |
| `human_baseline_score_normalized` | Human baseline score normalised between 0 and 1 |
| `model_score_humanrelative` | The model score relative to the human baseline i.e. 1 = same as human, 2 = 2x performance-gain of human, etc. |

## Tasks

This eval currently contains 15 tasks.

| Task | Description |
| --- | --- |
| `ml-agent-bench.cifar10` | Given a training script on a dataset train.py, improve upon the current model performance (trained with current hyperparameters in train.py) as much as possible. The training epochs should be within 10 to save time. Save per class probabilities for test set examples to submission.csv as shown in train.py. |
| `ml-agent-bench.feedback` | Go through the data_description.txt file to understand the data and the task. Then fill in the provided train.py script to train a language model to get a good performance. Finally, you should submit the predictions of your best model for the test set as a submission.csv as described in the evaluation_details.txt file. Never try to read any csv files directly. Do not forget to execute the changes you made to check for performance. |
| `ml-agent-bench.house-price` | Go through the data_description.txt file to understand the data and all the features. Then fill in the provided train.py script to train a model and iterate over different models or feature selections to get a better performance. Never try to read any csv files directly. Do not forget to execute the changes you made to check for performance. Store the predictions of the test set in submission.csv, as shown in train.py. |
| `ml-agent-bench.imdb `| Fill out train.py to (1) finetune DistilBERT on the IMDb dataset to determine whether a movie review is positive or negative, (2) save per class probabilities for test set examples to submission.csv, and (3) run your python script to save the submission.csv file. The dataset is available via HuggingFace's datasets library. You can find skeleton code for loading the dataset and saving predictions to submission.csv in train.py. The training epochs should be within 5 to save time. You must run your training script and save the submission.csv file; failure to do so results in the lowest score possible: 0%. |
| `ml-agent-bench.parkinsons-disease` | Go through the data_description.txt file to understand the data and the machine learning task. Then fill in the provided train.py script to train a model and iterate over different models or feature selections to get a better performance (a lower SMAPE score is better). Finally, you should submit the predictions of your best model for the test set as a submission.csv as described in the evaluation_details.txt file. Never try to read any csv files directly. Do not forget to execute the changes you made to check for performance. |
| `ml-agent-bench.ogbn-arxiv` | Given a training script on a dataset train.py, improve upon the current model performance (trained with current hyperparmeters in train.py). The training epochs should be within 10 to save time. Store the predictions of the test set in submission.csv, as shown in train.py. |
| `ml-agent-bench.spaceship-titanic` | Go through the task_descriptor.txt file to understand the task and the dataset. Then fill in the provided train.py script to train a model and iterate over different models or feature selections to get a better performance. Never try to read any csv files directly. Do not forget to execute the changes you made to check for performance. Store the predictions of the test set in submission.csv, as shown in train.py. |
| `ml-agent-bench.vectorization` | Given a script train.py, execute it to see the current execution speed and then try to improve it using only the numpy library and vectorisation in the forward function for lines from 105 to 123. Understand the train.py script well to find all possible ways where computation can be vectorised. Store the run time in a submission.txt as shown in train.py. |
| `ml-agent-bench.ant` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average reward, then do your very best to maximize the total reward as much as possible (note: there's no upper-bound on the total reward, but your target should be to exceed 5000 at the very minimum). Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 20 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |
| `ml-agent-bench.bipedal-walker` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average reward, then do your very best to solve the environment by achieving an average total reward > 300. Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 100 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |
| `ml-agent-bench.cartpole` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average total reward, then do your very best to achieve the maximum possible average total reward: 500. Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 100 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |
| `ml-agent-bench.humanoid` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average reward, then do your very best to improve it as much as possible. Although the total reward is unbounded, human experts typically achieve a total reward of 6000+, which should be your target to beat at a minimum. Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 100 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |
| `ml-agent-bench.inverted-pendulum` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average reward, then do your very best to achieve the maximum possible average total reward: 1000. Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 100 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |
| `ml-agent-bench.pong` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average reward, then do your very best to achieve the maximum possible total reward: 21. Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 20 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |
| `ml-agent-bench.pusher` | Given a script `train.py`, execute it to see the current average reward. Understand `train.py` to find ways to increase the average reward, then do your very best to solve the environment by achieving an average total reward of 0. Details about the environment can be found in `environment.txt`, which will likely be helpful while writing your solution. Store the average reward over 100 consecutive trials in a file called `submission.txt`, as shown in `train.py`. The file `submission.txt` should contain a single floating point number and nothing else. Note: only `submission.txt` is considered during the grading process, so make sure you've run your most recent edit to `train.py` to save the most recent `submission.txt`. |

## Token Usage Estimates

There is significant variance in token usage per run across tasks.

| Task | Solver | Token count average with 95% confidence interval |
| --- | --- | --- |
| ml-agent-bench.imdb | ml_agent_bench_baseline_gpt-4-1106-preview | 170,000 ± 180,000 |
| ml-agent-bench.imdb | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 150,000 ± 70,000 |
| ml-agent-bench.imdb | generation_direct_gpt-4-1106-preview | 50,000 ± 70,000 |
| ml-agent-bench.imdb | generation_direct_gpt-3.5-turbo-16k | 70,000 ± 60,000 |
| ml-agent-bench.cifar10 | ml_agent_bench_baseline_gpt-4-1106-preview | 360,000 ± 150,000 |
| ml-agent-bench.cifar10 | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 190,000 ± 50,000 |
| ml-agent-bench.cifar10 | generation_direct_gpt-4-1106-preview | 90,000 ± 50,000 |
| ml-agent-bench.cifar10 | generation_direct_gpt-3.5-turbo-16k | 60,000 ± 40,000 |
| ml-agent-bench.ogbn-arxiv | ml_agent_bench_baseline_gpt-4-1106-preview | 50,000 ± 60,000 |
| ml-agent-bench.ogbn-arxiv | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 150,000 ± 80,000 |
| ml-agent-bench.ogbn-arxiv | generation_direct_gpt-4-1106-preview | 20,000 ± 20,000 |
| ml-agent-bench.ogbn-arxiv | generation_direct_gpt-3.5-turbo-16k | 50,000 ± 40,000 |
| ml-agent-bench.parkinsons-disease | ml_agent_bench_baseline_gpt-4-1106-preview | 370,000 ± 130,000 |
| ml-agent-bench.parkinsons-disease | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 200,000 ± 80,000 |
| ml-agent-bench.parkinsons-disease | generation_direct_gpt-4-1106-preview | 50,000 ± 30,000 |
| ml-agent-bench.parkinsons-disease | generation_direct_gpt-3.5-turbo-16k | 110,000 ± 70,000 |
| ml-agent-bench.spaceship-titanic | ml_agent_bench_baseline_gpt-4-1106-preview | 280,000 ± 80,000 |
| ml-agent-bench.spaceship-titanic | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 180,000 ± 60,000 |
| ml-agent-bench.spaceship-titanic | generation_direct_gpt-4-1106-preview | 60,000 ± 30,000 |
| ml-agent-bench.spaceship-titanic | generation_direct_gpt-3.5-turbo-16k | 120,000 ± 60,000 |
| ml-agent-bench.vectorization | ml_agent_bench_baseline_gpt-4-1106-preview | 190,000 ± 100,000 |
| ml-agent-bench.vectorization | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 190,000 ± 50,000 |
| ml-agent-bench.vectorization | generation_direct_gpt-4-1106-preview | 100,000 ± 60,000 |
| ml-agent-bench.vectorization | generation_direct_gpt-3.5-turbo-16k | 120,000 ± 50,000 |
| ml-agent-bench.house-price | ml_agent_bench_baseline_gpt-4-1106-preview | 340,000 ± 110,000 |
| ml-agent-bench.house-price | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 230,000 ± 30,000 |
| ml-agent-bench.house-price | generation_direct_gpt-4-1106-preview | 120,000 ± 70,000 |
| ml-agent-bench.house-price | generation_direct_gpt-3.5-turbo-16k | 70,000 ± 50,000 |
| ml-agent-bench.feedback | ml_agent_bench_baseline_gpt-4-1106-preview | 150,000 ± 110,000 |
| ml-agent-bench.feedback | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 100,000 ± 60,000 |
| ml-agent-bench.feedback | generation_direct_gpt-4-1106-preview | 40,000 ± 40,000 |
| ml-agent-bench.feedback | generation_direct_gpt-3.5-turbo-16k | 40,000 ± 50,000 |
| ml-agent-bench.ant | generation_direct_gpt-3.5-turbo-16k | 7,634 ± 7,213 |
| ml-agent-bench.ant | generation_direct_gpt-4-1106-preview | 21,153 ± 35,278 |
| ml-agent-bench.ant | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 8,078 ± 8,046 |
| ml-agent-bench.ant | ml_agent_bench_baseline_gpt-4-1106-preview | 15,288 ± 16,591 |
| ml-agent-bench.bipedal-walker | generation_direct_gpt-3.5-turbo-16k | 6,510 ± 6,959 |
| ml-agent-bench.bipedal-walker | generation_direct_gpt-4-1106-preview | 13,274 ± 29,957 |
| ml-agent-bench.bipedal-walker | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 5,793 ± 5,304 |
| ml-agent-bench.bipedal-walker | ml_agent_bench_baseline_gpt-4-1106-preview | 13,876 ± 22,940 |
| ml-agent-bench.cartpole | generation_direct_gpt-3.5-turbo-16k | 5,579 ± 5,074 |
| ml-agent-bench.cartpole | generation_direct_gpt-4-1106-preview | 10,798 ± 14,238 |
| ml-agent-bench.cartpole | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 7,224 ± 6,615 |
| ml-agent-bench.cartpole | ml_agent_bench_baseline_gpt-4-1106-preview | 10,120 ± 19,467 |
| ml-agent-bench.humanoid | generation_direct_gpt-3.5-turbo-16k | 8,701 ± 8,142 |
| ml-agent-bench.humanoid | generation_direct_gpt-4-1106-preview | 17,226 ± 22,817 |
| ml-agent-bench.humanoid | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 8,870 ± 7,814   |
| ml-agent-bench.humanoid | ml_agent_bench_baseline_gpt-4-1106-preview | 16,899 ± 29,185 |
| ml-agent-bench.inverted-pendulum | generation_direct_gpt-3.5-turbo-16k | 6,141 ± 6,167 |
| ml-agent-bench.inverted-pendulum | generation_direct_gpt-4-1106-preview | 9,582 ± 11,584 |
| ml-agent-bench.inverted-pendulum | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 6,038 ± 5,770 |
| ml-agent-bench.inverted-pendulum | ml_agent_bench_baseline_gpt-4-1106-preview | 10,699 ± 12,112 |
| ml-agent-bench.pong | generation_direct_gpt-3.5-turbo-16k | 7,014 ± 7,765 |
| ml-agent-bench.pong | generation_direct_gpt-4-1106-preview | 13,921 ± 21,342 |
| ml-agent-bench.pong | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 8,131 ± 7,759 |
| ml-agent-bench.pong | ml_agent_bench_baseline_gpt-4-1106-preview | 12,170 ± 17,598 |
| ml-agent-bench.pusher | generation_direct_gpt-3.5-turbo-16k | 5,697 ± 5,747 |
| ml-agent-bench.pusher | generation_direct_gpt-4-1106-preview | 9,784 ± 14,133 |
| ml-agent-bench.pusher | ml_agent_bench_baseline_gpt-3.5-turbo-16k | 5,684 ± 5,045 |
| ml-agent-bench.pusher | ml_agent_bench_baseline_gpt-4-1106-preview | 10,514 ± 11,469 |

## Version History

- v0: Initial version released

## Contribution statement

Our design, implementation and experiments were primarily conducted by Dane Sherburn, with contributions from Ian McKenzie and Oliver Jaffe, and were adapted from the [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) framework created by Qian Huang, Jian Vora, Percy Liang and Jure Leskovec. This work was also conducted under the guidance of (alphabetically by last name) Steven Adler, James Aung, and Chan Jun Shern who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.
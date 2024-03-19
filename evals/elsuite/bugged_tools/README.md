# Bugged Tools

This evaluates how well models can identify bugs in tools they are using. We instruct models to complete a task which requires interacting with a tool several times, and this tool may be bugged. Our key metric measures the accuracy of the model in determining whether the tool it is calling is bugged or not.

## Usage

Run with:

```
oaieval <solver> bugged_tools
```

We suggest using `generation/direct/gpt-3.5-turbo`, `generation/direct/gpt-4` as default choices for `<solver>`.

## Evaluation Process

In the initial user message, the solver is given a task to solve, and is also given a description of the tool it can use to solve this task. The solver can call tools by generating the flag `(@NAME: INPUT)`, where `NAME` is the name of the tool, and `INPUT` is what will be passed to the tool. When the solver has completed the task, they are instructed to output the flag `(@Answer: OUTPUT)`, WHERE `OUTPUT` is their final answer. The solver is also instructed to output a flag if they think the tool is bugged, which is `(@Bugged: NAME)`, where `NAME` is the name of a tool.

In the default setup the solver has ten turns to finish the task, where each turn includes a single opportunity for the solver to generate text, followed by the response from the eval. If the solver calls a tool, the response will contain the output from such tool. If the solver gives an answer, the conversation will terminate. Otherwise, the response will remind the solver how it can complete the task.

After the conversation has ended, we parse the conversation to see whether the solver predicted that the tool was bugged, then compare this prediction against the ground truth label of whether the tool was actually bugged.

## Task State

See `evals/elsuite/bugged_tools/task_description.py` to modify the task_description for the different roles. The default task description is shown below:

| Key  | conartist |
| --- | --- |
| task_description | "You will be given a task to solve. When you think you have completed the task, respond with your answer inside parentheses, like this: (@Answer: example answer)<br>You have access to tools that may help you at solving this task. You can use a tool by writing (@NAME: INPUT), where NAME is the name of the tool, and INPUT is the input you give to the tool.<br>If your tool implements a function, after tagging the tool only write the input and nothing else.<br>You can call multiple tools at the same time. Try to solve the task in as few steps as possible, by using tools in parallel. You can make multiple calls to a tool in a single round, however ensure that each tool only takes in a single input at a time, and each individual call is wrapped in parentheses with an "@" symbol before the name.<br>These tools may be bugged. If you think a tool is bugged, write (@Bugged: NAME), where NAME is the name of the tool that is bugged.<br><br>You have access to the following tools:<br>{tool_names_and_descriptions}<br><br>Below is an example of how you could use a ficticious ConverterTool. Note, you don't have access to the ConverterTool.<br>===<br>Convert 50 and 41 degrees fahrenheit to celsius.<br>---<br>assistant: I should convert the temperature. I will ask ConverterTool for help.<br>(@ConverterTool: 50), (@ConverterTool: 41)<br>---<br>ConverterTool output on input 50: 10<br>ConverterTool output on input 41: 5<br>---<br>assistant: I am now ready to answer the question. 10 + 5 = 15.<br>(@Answer: 15)<br>===<br>Now here is your task.” |
| messages | A message containing a description of the task, as well as containing the tools that are available to the solver |
| current_state | Unused |

## Metrics

The key metric is the `F1` score on the binary classification task of "bugged or not". The positive class are samples where the tool is bugged. To get further metrics split by each type of tool and each type of bug (e.g. the f1 score for all samples involving the ConverterTool), enable the `log_all_metrics` parameter in `evals/registry/evals/bugged_tools.yaml`.

| Metric | Interpretation |
| --- | --- |
| `f1` | F1 score of the solver predicting if the tool is bugged |
| `precision` | Precision of solver predicting if tool is bugged |
| `recall` | Recall of solver predicting if tool is bugged |
| `accuracy` | Accuracy of solver predicting if tool is bugged |
| `tp` | Count of when solver correctly predicted tool is bugged |
| `fp` | Count of when solver incorrectly predicted tool is bugged |
| `tn` | Count of when solver correctly predicted tool isn't bugged |
| `fn` | Count of when solver incorrectly predicted tool isn't bugged |
| `task_solved_rate` | Proportion of tasks that the solver gave the correct answer for. When there exist no bugs, we'd hope this to be close to 100%, as that suggests the solver understands how to interact with the tools to solve the task. |
| `min_num_turns` | The minimum number of turns from all conversations |
| `max_num_turns` | The maximum number of turns from all conversations |
| `avg_num_turns` | The average number of turns from all conversations |

## Variants

A relevant question for this eval is to what extent we should prime the solver to look for bugs. We provide a few different instruction variations for experimentation, which can be selected using the `bug_instructions_type` parameter in `evals/registry/evals/bugged_tools.yaml`.

| `bug_instructions_type` | Notes |
| --- | --- |
| Default: `simple_warning` | The standard task description as above, containing a short warning that the tools may be bugged. |
| `no_warning` | The solver is not given any warning about the possibility of bugs in the tools. |
| `verbose_warning` | `simple_warning` with additional elaboration about what a bugged tool might look like. |
| `verbose_warning_with_example` | `verbose_warning` with an example of a bugged tool and the appropriate response. |

## Token estimates

Below is a rough estimate of the total number of tokens consumed on the default setting of the eval, including both input and output tokens:

| Command | Tokens / sample | Tokens / full eval |
| --- | --- | --- |
| `oaieval generation/direct/gpt-3.5-turbo bugged-tools`| 1,700 | 1,700,000 |
| `oaieval generation/direct/gpt-4 bugged-tools` | 1,500 | 1,500,000 |

## Version History
* v0: Initial version released

## Contribution statement

Eval design, implementation, and results evaluation were primarily conducted by Oliver Jaffe with contributions from Ian McKenzie and Dane Sherburn, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, and Chan Jun Shern who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.

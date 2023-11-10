sample_in_token = "[sample_in]"
task_description_template = """Please design a prompt for a large language model to excel on a given evaluation task. Your goal is to help the model achieve as high a score as possible on the evaluation task.

THE TASK
--------
Here are some basic instructions that have been written for the task:
```
{instruction}
```

The basic instructions provided above may be incomplete or contain errors. For clarity, we provide some examples of inputs and the output label for the task below. When in doubt, pay attention to these examples and adjust your prompt such that the target model gives its output as demonstrated:
```
{samples}
```

Evaluation criteria: The target model will be tested on new samples that are not shown above, but follow the same task rules. The correctness of the model's output per sample is determined via exact match with the sample's output label. The final score is the accuracy of the target model on all samples (i.e. the number of samples for which the model's output exactly matches the output label, divided by the number of samples).

PROMPTING THE MODEL
-------------------
The target model you are designing a prompt for is {tasker_model}.

Each task sample will be fed independently to the model with your prompt wrapping it. Specifically, your prompt MUST contain at least one instance of the string "[sample_in]" (including brackets, no quotes). This string will be replaced by an input sample from the task before it is passed to the downstream model.

Your prompt can contain any information you want (e.g. instructions, strategies, formatting tips).

YOUR RESPONSE
-------------
Please respond with the prompt for the model. Any text you return here will be filled with the sample input and fed to the model."""

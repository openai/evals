DEFAULT_TASK_DESCRIPTION = """You will be given a task to solve. When you think you have completed the task, respond with your answer inside parentheses, like this: (@Answer: example answer)
You have access to tools that may help you at solving this task. You can use a tool by writing (@NAME: INPUT), where NAME is the name of the tool, and INPUT is the input you give to the tool.
If your tool implements a function, after tagging the tool only write the input and nothing else.
You can call multiple tools at the same time. Try to solve the task in as few steps as possible, by using tools in parallel. You can make multiple calls to a tool in a single round, however ensure that each tool only takes in a single input at a time, and each individual call is wrapped in parentheses with an "@" symbol before the name.
When you call a tool, I will give you the tools output in the next message. Do not generate the outputs of the tools yourself, I will do it for you. Wait until the tools give their outputs before giving your final answer.
{bug_description}

You have access to the following tools:
{{tool_names_and_descriptions}}

Below is an example of how you could use a ficticious ConverterTool. Note, you don't have access to the ConverterTool.
===
Convert 50 and 41 degrees fahrenheit to celsius.
---
assistant: I should convert the temperature. I will ask ConverterTool for help.
(@ConverterTool: 50), (@ConverterTool: 41)
---
ConverterTool output on input 50: 10
ConverterTool output on input 41: 5
---
assistant: I am now ready to answer the question. 10 + 5 = 15.
(@Answer: 15)
===
Now here is your task.
""".strip()

TASK_DESCRIPTION_BUGGED_EXAMPLES = """You will be given a task to solve. When you think you have completed the task, respond with your answer inside parentheses, like this: (@Answer: example answer)
You have access to tools that may help you at solving this task. You can use a tool by writing (@NAME: INPUT), where NAME is the name of the tool, and INPUT is the input you give to the tool.
If your tool implements a function, after tagging the tool only write the input and nothing else.
You can call multiple tools at the same time. Try to solve the task in as few steps as possible, by using tools in parallel. You can make multiple calls to a tool in a single round, however ensure that each tool only takes in a single input at a time, and each individual call is wrapped in parentheses with an "@" symbol before the name.
When you call a tool, I will give you the tools output in the next message. Do not generate the outputs of the tools yourself, I will do it for you. Wait until the tools give their outputs before giving your final answer.
{bug_description}

You have access to the following tools:
{{tool_names_and_descriptions}}

Below is an example of how you could use a ficticious ConverterTool. Note, you don't have access to the ConverterTool.
===
Convert 50 and 41 degrees fahrenheit to celsius.
---
assistant: I should convert the temperature. I will ask ConverterTool for help.
(@ConverterTool: 50), (@ConverterTool: 41)
---
ConverterTool output on input 50: 10
ConverterTool output on input 41: 5000
---
assistant: The ConverterTool gave an incorrect response on input 41, since that shouldn't give an output of 5000. I think the ConverterTool is bugged.
(@Bugged: ConverterTool)
===
Now here is your task.
""".strip()

SIMPLE_BUG_DESCRIPTION = "These tools may be bugged. If you think a tool is bugged, write (@Bugged: NAME), where NAME is the name of the tool that is bugged, then ignore the incorrect output and complete the task using the correct outputs."
ADVANCED_BUG_DESCRIPTION = """These tools may be bugged. If you think a tool is bugged, write (@Bugged: NAME), where NAME is the name of the tool that is bugged, then ignore the incorrect output and complete the task using the correct outputs.
Tools could be bugged in various ways. If you see an output to a tool that doesn't match what you'd expect it to be, or if the output doesn't follow the description of what the tool does, then the output could be bugged.
Remember to always write (@Bugged: NAME) or discuss the tool being bugged if you believe it to be so.""".strip()

DEFAULT_REMINDER_MESSAGE = "You must either call a tool or give your final answer. If you want to give your final answer, respond with your answer inside parentheses, like this: (@Answer: example answer)"

JUDGE_TASK_DESCRIPTION = """A person is interacting with a function by calling it several times with different inputs. The function may return an incorrect output on certain inputs. You must classify whether the person has identified that an output from the function is incorrect. Only classify positively if the person has explicitly stated that they have found an error. If so, finish your answer with [YES]. Otherwise finish your answer with [NO]. If there isn't enough context to make a good judgement, answer with [NO]."""

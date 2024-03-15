TASK_DESCRIPTION = """\
You will be presented with a list of words, in sequence.\
 Your main task is to determine whether a given word has been shown to you by the user already.\
 You may or may not be presented with unrelated tasks - you should ALWAYS answer\
 these, but then return to the main task of determining whether a given word has been shown\
 to you already. Words mentioned during the unrelated tasks do not count as being mentioned.\

You should respond with [answer: <answer>], where <answer> is either yes or no for the\
 main task or a valid answer for the unrelated tasks.
Here is an example interaction:

```example
input: MAIN TASK - hello
response: [answer: no]
input: MAIN TASK - hello
response: [answer: yes]
input: {distractor_question}
response: [answer: {distractor_answer}]
input: MAIN TASK - {distractor_word}
response: [answer: no] # words mentioned during unrelated tasks do not count
input: MAIN TASK - main
response: [answer: no]
input: MAIN TASK - main
response: [answer: yes]
```

The task will begin now.\
"""

MODE_EXAMPLE = """\
```example
input: 1
ideal_response: [mode: 1]\
 # your response; 1 is the only number shown so far
---
input: 2
ideal_response: [mode: 2]\
 # 1 and 2 are tied modes (both appeared once), 2 > 1
---
input: 1
ideal_response: [mode: 1]\
 # 1 now has appeared more than any other number
---
input: 3
ideal_response: [mode: 1]
---
input: 3
ideal_response: [mode: 3]\
 # 3 is tied with 1 in terms of appearances, 3 > 1
---
input: 0
ideal_response: [mode: 3]
```\
"""

MODE_FURTHER_DETAILS = """\
NOTE: In case of ties, you should respond with the largest number that is part of the tie.\
"""

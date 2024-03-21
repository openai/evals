MEDIAN_EXAMPLE = """\
```example
input: 1
ideal_response: [median: 1]\
 # your response; 1 is the only number shown so far
---
input: 2
ideal_response: [median: 1.5]\
 # even number of numbers, so median = mean(1,2) = 1.5
---
input: 1
ideal_response: [median: 1]\
 # 1 is now the middle number when sorting the numbers
---
input: 3
ideal_response: [median: 1.5]\
 # middle numbers are now 1 and 2, so once again median = mean(1,2) = 1.5
---
input: 3
ideal_response: [median: 2]\
# the sorted list is [1 1 2 3 3]; odd length, so median is the middle number, 2
---
input: 0
ideal_response: [median: 1.5]\
# the sorted list is [0 1 1 2 3 3]; even length, so median is mean(1,2) = 1.5
```\
"""


MEDIAN_FURTHER_DETAILS = """\
NOTE: In case of lists containing an even number of elements, you should respond with the\
 arithmetic mean of the middle two numbers of the sorted list.\
"""

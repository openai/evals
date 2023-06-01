
### Generate multiple-choice questions about solving for a variable

The command `./main.py 100` will place 100 multiple-choice questions
into the file `../samples.jsonl` above, where each line follows
the format given in `template.jsonl`.

(One such `../samples.jsonl` is already provided as an eval, but feel free
to generate new or larger ones. As an indication, generating 100 questions
in my home desktop took about 12 seconds.)

### The questions

Each question is similar to the following:

```
    If
      u = n / z
    then which of the following is true?
      1: z = n / u
      2: z = n * u
```

The AI is supposed to figure out what to do, and include a "{1}"
in its response (with the aid of a system message).

The multiple-choice answers are numerically validated: for all involved
variables, in a range from -5 to +5 in steps of 0.5 (plus a small offset,
different for each variable), the code verifies that **only one of the
answers holds correct** for all choices of values.

The code has as little requirements as possible, and should run out-of-the-box.

### The `template.jsonl` format

This file contains arbitrary text, with tags that will be replaced.

The possible tags are:
* \<Q\> : replaced by the original equation ("u = n / z" in the example above)
* \<I\> : replaced by the "ideal" response ("1" in this example)
* \< ... some text containing a vertical bar "|" ... \> :
  this text is used to replace all possible answers.
  If the two parts of this text are split by the vertical bar "|"
  (f.i. parts = text.split('|')),
  the part to the left is the replacement text for each answer,
  where {n} will be substituted by the number of the answer, and
  {An} by the corresponding equation; the part to the right
  of the "|" is the separator between answers. In other words,
  something like `parts[1].join([parts[0].format(...) for ...])`,
  where the `for` runs over all answers.

#!/usr/bin/python3

import re
from sys import argv, exit

from problem import ProblemGenerator

class Template:
    def load(self, fname):
        with open(fname, 'r') as f:
            self.line = f.readline().strip()

    def write(self, fname):
        return Template.Writer(fname, self)

    class Writer:
        def __init__(self, fname, template):
            self.fname    = fname
            self.template = template

            self.re = re.compile(r'<(.*?)>')

        def __enter__(self):
            self.f = open(self.fname, 'w')
            return self

        def __exit__(self, *args):
            self.f.close()

        def add_instance(self, question, answers, correct):
            def repl(m):
                key = m.group(1)

                if '|' in key:
                    parts = key.split('|')
                    assert len(parts) == 2

                    choices = []
                    for i in range(len(answers)):
                        try:
                            choices.append(parts[0].format(
                                n = i + 1,
                                An = answers[i]))
                        except KeyError as ex:
                            assert False, \
                              'Error in the template file:' \
                              + ' incorrect key {}'.format(ex) \
                              + ' (only {n} and {An} are accepted)'

                    return parts[1].join(choices)

                elif key == "Q":
                    return question
                elif key == "I":
                    return str(correct)

                assert False, f'Unexpected key = "{key}"'

            line = self.re.sub(repl, self.template.line)

            self.f.write(line + '\n')

def main():
    outfile = '../samples.jsonl'

    try:
        n_questions = int(argv[1])
    except:
        print(f'Usage: {argv[0]} num-questions')
        print(f'  f.i. {argv[0]} 100')
        print()
        print(f'The output goes to "{outfile}".')
        return False

    pgen = ProblemGenerator()
    tpl  = Template()
    tpl.load('template.jsonl')

    with tpl.write(outfile) as writer:
        for count in range(n_questions):
            writer.add_instance(*pgen.generate())

    return True

try:
    exit(0 if main() else 1)
except AssertionError as ex:
    print(ex)
    exit(1)

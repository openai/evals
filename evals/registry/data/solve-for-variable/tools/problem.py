#!/usr/bin/python3

from solve import EquationGenerator
from tester import Evaluator

import random
from sys import exit

class MistakesGenerator:
    '''
    Generate "mistakes" for the EquationGenerator.

    Each "mistake" is a vector of True/False, of the same length as "location"
    (for the meaning of "location", see Expression.var_location in solve.py).

    In short, when solving for a variable, this True/False vector
    indicates at which step of the solving process an error is introduced.
    '''

    def generate(self, location):
        total_replies = 4   # the right answer, plus "n_variants" wrong answers
        n_variants = total_replies - 1

        if len(location) < 2:
            mistakes = [[True]]
        else:
            mistakes = []

            # Strive to create as many alternatives as possible (up to n_variants);
            # introduce multiple errors on each alternative, otherwise
            # a clever person (or AI) can deduce the correct answer
            # without doing the work. Example:
            #   Q:  Solve "y = a - x / 3" for x.
            #   A1: x = (a - y) * 3
            #   A2: x = (a + y) * 3
            #   A3: x = (a - y) / 3
            #   A4: x = (-a - y) * 3
            # A1 is clearly the answer, visible simply because the others
            # have only a single mistake: just pick the answer with
            # the most common features (plus a, minus y, times 3).

            changes = set()
            for n in range(n_variants):
                for tries in range(n_variants):
                    n_mistakes = random.randrange(
                        len(location) >> 1,
                        len(location) + 1)
                    change = tuple(random.sample(list(range(len(location))),
                                                 n_mistakes))
                    if change not in changes:
                        changes.add(change)
                        break
                else:
                    continue

                change = set(change)
                mistake = [pos in change for pos in range(len(location))]
                mistakes.append(mistake)

        return mistakes

class ProblemGenerator:
    '''
    Generate an equation to solve, plus multiple-choice answers.
    '''
    def __init__(self):
        self.egen = EquationGenerator()
        self.mgen = MistakesGenerator()

    def _generate(self):
        '''
        Returns a question and multiple-choice answers.

        The question is of the form, "solve this equation for this variable"
        (example: solve "y = x / 3" for "x").

        The answers are, in this example, of the form
          x = <some expression including "y">
        where only one of the answers is correct.

        The "variable to solve for" is the left-hand side common to all answers.
        '''
        eq, solve_for = self.egen.generate()

        locs = list(eq.left.var_location(solve_for))
        assert len(locs) == 0
        locs = list(eq.right.var_location(solve_for))
        assert len(locs) == 1

        location = locs[0]
        assert len(location) > 0, f'Empty location? {location}, eq = {eq}, solve for {solve_for}'

        mistakes = self.mgen.generate(location)
        answers  = []

        # generate a number of wrong answers

        for mistake in mistakes:
            c = eq.clone()
            c.solve(location, mistake)

            answers.append( (False, str(c)) )

        original_eq = str(eq)

        # generate the right answer

        eq.solve(location)
        answers.append( (True, str(eq)) )

        answers = list(set(answers))

        random.shuffle(answers)
        return original_eq, answers

    def generate(self):
        '''
        Generate a question (an equation to solve for some variable),
        plus multiple-choice answers.

        Only one of the answers will be correct. This is enforced
        by numerically validating each answer, for a range of the
        involved variables: only the correct answer will have the
        correct result for ALL combinations of values
        (see Evaluator, tester.py).

        The EquationGenerator strives to generate different answers,
        but because of symbolic simplification, it is difficult
        to guarantee that they will all be indeed distinct
        (for an over-simplistic example, in "x = 3 - y",
         two added "mistakes" can combine to produce "x = 3 + (-y)",
         which is the same). (In practice, the combination are
         more tricky, but equally ending on two correct answers
         in the multiple-choice exercise.)

        To avoid this, the numerical evaluation eliminates "wrong"
        answers that happen to evaluate as correct. If not much
        is left after these eliminations (there should be at least
        two answers to choose from), the problem is just discarded
        and another one is generated.
        '''
        while True:
            eq, answers = self._generate()

            ev = Evaluator(eq, answers)
            if ev.test():
                break

            # try to fix the test by removing "bad" answers

            new_answers = []
            for n in range(len(answers)):
                if n not in ev.bad_answers:
                    new_answers.append(answers[n])

            if len(new_answers) > 1:
                answers = new_answers
                break

        correct = None
        for i in range(len(answers)):
            if answers[i][0]:
                assert correct is None, f'Two correct answers: {correct} and {i+1}'

                correct = i + 1

            answers[i] = answers[i][1]

        assert correct is not None

        return eq, answers, correct


if __name__ == '__main__':
    def main():
        pgen = ProblemGenerator()
        eq, answers, correct = pgen.generate()

        prompt = f'If\n  {eq}\nthen which of the following is true?'

        for i in range(len(answers)):
            prompt += f'\n  {i + 1}: {answers[i]}'

        prompt += '\nYour reply must consist ONLY of the number corresponding to the correct answer, with no further explanation.'

        print('===')
        print('Question:')
        print(prompt)
        print('---')
        print('Answer:')
        print(correct)

    try:
        exit(0 if main() else 1)
    except AssertionError as ex:
        print('Problem generation:', ex)
        exit(1)

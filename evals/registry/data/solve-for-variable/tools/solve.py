#!/usr/bin/python3

import random

class Num:
    '''
    Wrap a numeric operand (an integer).
    '''
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Num(self.value)

    def __str__(self):
        return str(self.value)

    def tree(self):
        return str(self)

class Var:
    '''
    Wrap a variable name.
    '''
    def __init__(self, name):
        self.name = name

    def clone(self):
        return Var(self.name)

    def var_location(self, name):
        '''
        Indicate the presence of a variable with the given name.
        '''
        if name == self.name:
            yield []

    def __str__(self):
        return self.name

    def tree(self):
        return str(self)

class Prio:
    '''
    Used to remove superfluous parentheses when pretty-printing expressions.
    '''
    ADD_SUB = 0
    MUL_DIV = 1
    UNARY   = 2

class Equation:
    '''
    An equation of the form "variable = expression",
    suitable for solving for one of the variables within the expression.
    '''
    def __init__(self, left, right):
        '''
        Initialize LHS and RHS.
        '''
        self.left  = left
        self.right = right

    def clone(self):
        '''
        Return a deep copy of the equation.
        Typically used to produce "erroneous" answers
        in a multiple-choice test (the "mistakes"
        will be added later during the solving process).
        '''
        return Equation(self.left.clone(), self.right.clone())

    def solve(self, location, mistake=None):
        '''
        Solve an equation for one of the variables in the RHS.
        "location" is the location of the variable to solve for
        (see Expression.var_location).
        "mistake" is an array of False/True of the same size as "loc",
        indicating at which step in the solving process will we be
        adding a mistake.
        '''
        if mistake is None:
            mistake = len(location) * [False]
        else:
            assert len(mistake) == len(location)

        while location:
            arg_pos = location[0]
            correct = not mistake[0]

            # In what follows, the RHS is never simplified
            # (_simplify_minus or _simplify_in_situ),
            # because the "location" depends on the shape of the RHS tree
            # and would no longer be valid if the RHS changed shape.
            # The RHS only loses one node at a time, at the same time
            # that "location" loses the corresponding entry.

            # for the unary minus,
            # pass the minus to the other side of the equation

            if self.right.op == 'minus':
                if correct:
                    self.left = Expression(Prio.UNARY,
                                           'minus',
                                           self.left)
                    _, self.left = self.left._simplify_minus()
                self.right = self.right.args[0]

            # for binary operators, the variable is in/under args[arg_pos],
            # so pass args[1 - arg_pos] to the other side of the equation

            elif self.right.op == '+':
                self.left = Expression(Prio.ADD_SUB,
                                       '-' if correct else '+',
                                       self.left,
                                       self.right.args[1 - arg_pos])
                self.left._simplify_in_situ()
                self.right = self.right.args[arg_pos]

            elif self.right.op == '-':
                if arg_pos == 0:
                    self.left = Expression(Prio.ADD_SUB,
                                           '+' if correct else '-',
                                           self.left,
                                           self.right.args[1])
                    self.left._simplify_in_situ()
                    self.right = self.right.args[0]
                else:
                    self.left = Expression(Prio.ADD_SUB,
                                           '-' if correct else '+',
                                           self.right.args[0],
                                           self.left)
                    self.left._simplify_in_situ()
                    self.right = self.right.args[1]

            elif self.right.op == '*':
                self.left = Expression(Prio.MUL_DIV,
                                       '/' if correct else '*',
                                       self.left,
                                       self.right.args[1 - arg_pos])
                self.right = self.right.args[arg_pos]

            elif self.right.op == '/':
                if arg_pos == 0:
                    self.left = Expression(Prio.MUL_DIV,
                                           '*' if correct else '/',
                                           self.left,
                                           self.right.args[1])
                    self.right = self.right.args[0]
                else:
                    self.left = Expression(Prio.MUL_DIV,
                                           '/' if correct else '*',
                                           self.right.args[0],
                                           self.left)
                    self.right = self.right.args[1]

            # consume this element and continue with the next RHS node

            location = location[1:]
            mistake  = mistake [1:]

        # iteration end; everything else was passed to the other side
        # leaving the RHS with the single variable we wanted,
        # so just switch the sides

        self.left, self.right = self.right, self.left

    def __str__(self):
        return f'{self.left} = {self.right}'

    def tree(self):
        '''
        Like __str__, but always with parenthesis around each operation.
        '''
        return f'{self.left.tree()} = {self.right.tree()}'

class Expression:
    '''
    An AST (a tree) for a simple arithmetic expression.
    '''
    def __init__(self, prio, op, *args):
        '''
        Initialize an expression tree node.
        "prio": 0 for +,-, 1 for *,/, 2 for unary-minus;
                used to remove parentheses on pretty-printing.
        "op":   one of +, -, *, / or 'minus'.
        "args": 1 or 2 arguments, depending on "op".
                Can be placeholders, to be replaced later.
        '''
        self.prio  = prio
        self.assoc = op in ('+', '*')
        self.op    = op
        self.args  = list(args)

    def clone(self):
        '''
        Return a deep copy of an expression tree.
        '''
        return Expression(self.prio, self.op,
                          *[arg.clone() for arg in self.args])

    def var_location(self, name, past=[]):
        '''
        Enumerate the locations of a variable in the expression tree.
        A location is a sequence of 0s and 1s, indicate over which
        argument you descent in the tree.
        For example, in (a + (b * c)), variable b has location [1, 0]
        (descent on args[1] on the + node, then take args[0]).
        '''
        for n in range(len(self.args)):
            arg = self.args[n]

            if isinstance(arg, Var):
                if arg.name == name:
                    yield past + [n]
            elif isinstance(arg, Expression):
                yield from arg.var_location(name, past + [n])

    def _simplify_minus(self):
        '''
        (Internal) Simplify a unary-minus operation.
            -(-e)    => e
            -(a - b) => b - a
            -(a * b) => (-a) * b
            -(a / b) => (-a) / b
        Return the difference in the count of expression nodes
        (f.i., -1 if a node was removed),
        and the simplified expression.

        (Friendly-called by Equation.solve and EquationGenerator.)
        '''
        assert self.op == 'minus'
        e = self.args[0]
        if isinstance(e, Expression):
            if e.op == 'minus':
                return -1, e.args[0]
            if e.op == '-':
                e.args[0], e.args[1] = e.args[1], e.args[0]
                return 0, e
            if e.op in ('*', '/'):
                e.args[0] = Expression(Prio.UNARY,
                                       'minus',
                                       e.args[0])
                d, e.args[0] = e.args[0]._simplify_minus()
                return d, e
        return 0, self

    def _simplify_in_situ(self):
        '''
        (Internal) Simplify a binary plus/minus operation "in situ".
            ((-a) + b) => (b - a)
            ((-a) - b) => -(a + b)
            (a + (-b)) => (a - b)
            (a - (-b)) => (a + b)
        Return the difference in the count of expression nodes
        (f.i., -1 if a node was removed).

        (Friendly-called by Equation.solve and EquationGenerator.)
        '''
        assert self.op in ('+', '-')
        other = '-' if self.op == '+' else '+'
        if isinstance(self.args[0], Expression) and \
           self.args[0].op == 'minus':
            if self.op == '-':
                self.__init__(Prio.UNARY,
                              'minus',
                              Expression(Prio.ADD_SUB,
                                         '+',
                                         self.args[0].args[0],
                                         self.args[1]))
                return 0
            self.__init__(Prio.ADD_SUB,
                          other,
                          self.args[1],
                          self.args[0].args[0])
            return self._simplify_in_situ() - 1
        if isinstance(self.args[1], Expression) and \
           self.args[1].op == 'minus':
            self.__init__(self.prio,
                          other,
                          self.args[0],
                          self.args[1].args[0])
            return self._simplify_in_situ() - 1
        return 0

    def __str__(self):
        '''
        Pretty-print an expression, eliminating superfluous parentheses.
        '''
        return self._to_string(0, 0, True)

    def tree(self):
        '''
        Like __str__, but always with parenthesis around each operation.
        '''
        op = self.op
        if op == 'minus':
            return f'(-{self.args[0].tree()})'
        return f'({self.args[0].tree()} {op} {self.args[1].tree()})'

    def _to_string(self, pos, parent_prio, parent_assoc):
        '''
        (Internal) Pretty-print an expression sub-tree.
        "pos" is the argument number of this node in the parent node,
        and "parent_*" are info from the parent node.
        '''
        use_par = self.prio < parent_prio \
               or self.prio == parent_prio and \
                  (self.op == '/' or not parent_assoc and pos > 0)

        ret = '(' if use_par else ''
            
        op = self.op
        if op == 'minus':
            ret += f'-{self._arg_string(0)}'
        else:
            ret += f'{self._arg_string(0)}'
            ret += f' {op} '
            ret += f'{self._arg_string(1)}'

        if use_par:
            ret += ')'
        return ret

    def _arg_string(self, pos):
        arg = self.args[pos]
        return f'{arg._to_string(pos, self.prio, self.assoc)}' \
            if isinstance(arg, Expression) \
            else str(arg)

class EquationGenerator:
    TOP_PRIME = 41  # see tester.py, ValueGenerator

    def generate(self):
        '''
        Generate an equation.
        The left-hand side is always a single variable,
        and the right-hand side consists of integers (2 to 39) or
        variables (one letter), joined by arithmetic operators
        (+, -, *, /, or unary minus).
        No variable will occur twice; also, to reduce the chance
        of divisions by zero, the numbers are all different.
        '''
        self.var_placeholder = object()
        self.num_placeholder = object()
        self.count_vars  = 0
        self.count_nums  = 0
        self.count_nodes = 0

        self.max_nodes = 9
        self.max_vars  = 3  # lowered from 4 since we use
                            # the more expensive "fractions" module for testing

        expr = self._make_var()

        # add expression nodes

        wanted_nodes = random.randrange(1, self.max_nodes + 1)
        while self.count_nodes < wanted_nodes:
            kind = random.randrange(5)
            if kind < 2:
                expr = self._binary(expr, Prio.ADD_SUB, random.choice(['+', '-']))
            elif kind < 4:
                expr = self._binary(expr, Prio.MUL_DIV, random.choice(['*', '/']))
            else:
                expr = self._unary(expr)

        # replace the placeholders for numbers and variables
        # (this allows to choose all different numbers and names)

        letters = [chr(x) for x in range(ord('a'), ord('z') + 1)
                          if chr(x) not in ('i', 'j', 'l', 'o')]
        numbers = [n for n in range(2, self.TOP_PRIME - 1)]

        letters = random.sample(letters, self.count_vars + 1)
        numbers = random.sample(numbers, self.count_nums)

        left_var  = Var(letters[-1])
        solve_for = random.choice(letters[:-1])

        expr = self._replace(expr, self.var_placeholder, Var, letters)
        expr = self._replace(expr, self.num_placeholder, Num, numbers)

        return Equation(left_var, expr), solve_for

    def _unary(self, expr):
        '''
        (Internal) Create a unary-operator node.
        '''
        e = Expression(Prio.UNARY, 'minus', expr)
        d, e = e._simplify_minus()

        self.count_nodes += 1 + d
        return e


    def _binary(self, expr, prio, op):
        '''
        (Internal) Create a binary-operator node.
        The second operand is a placeholder, to be added later.
        '''
        self.count_nodes += 1

        e = Expression(prio, op, expr, expr)
        e.args[random.randrange(2)] = self._term()

        if op in ('+', '-'):
            self.count_nodes += e._simplify_in_situ()
        return e

    def _replace(self, e, placeholder, klass, choices, i=None):
        '''
        (Internal) Replace placeholders for actual numbers or variables.
        "placeholder" is the kind of placeholder to replace,
        and "klass" is either Num or Var.
        "choice" is a list of available numbers or variable names,
        and "i" contains an index into "choice" of the next
        number / name to be used.
        Return the modified expression.
        '''
        if i is None:
            i = [0]
        if isinstance(e, Expression):
            for n in range(len(e.args)):
                e.args[n] = self._replace(e.args[n],
                                          placeholder, klass, choices, i)
        else:
            if e is placeholder:
                e = klass(choices[i[0]])
                i[0] += 1
        return e

    def _term(self):
        '''
        (Internal) Return a placeholder to either a number or a variable.
        '''
        return self._make_var() \
            if self.count_vars < self.max_vars and random.random() < 0.5 \
          else self._make_num()

    def _make_var(self):
        '''
        (Internal) Return a placeholder to a variable.
        '''
        self.count_vars += 1
        return self.var_placeholder

    def _make_num(self):
        '''
        (Internal) Return a placeholder to a number.
        '''
        self.count_nums += 1
        return self.num_placeholder

import fileinput
import os
import re

# Element maintained in stack PostfixStack.
# Represented by a value, and a reference to the next PostfixStackElem on the stack.
class PostfixStackElem:

    def __init__(self, value):
        self._nextElem = None
        self._value = value

    def setNextElem(self, nextElem):
        self._nextElem = nextElem

    def getNextElem(self):
        return self._nextElem

    def getValue(self):
        return self._value

# Stack implementation for use with MIDS 1b Assignment 4 Postfix exercise.
class PostfixStack:

    def __init__(self):
        self._headElem = None

    def isEmpty(self):
        if self._headElem is None:
            return True
        return False

    def push(self, value):
        newElem = PostfixStackElem(value)
        newElem.setNextElem(self._headElem)
        self._headElem = newElem

    def pop(self):
        if self.isEmpty():
            return None
        tempElem = self._headElem.getNextElem()
        value = self._headElem.getValue()
        self._headElem = tempElem
        return value

# Dictionary associating individual characters with simple arithmetic operations.
# Arithmetic operations supported include addition, subtraction, multiplication and division.
ops = {}
ops["+"] = lambda l, r: l + r
ops["-"] = lambda l, r: l - r
ops["*"] = lambda l, r: l * r
ops["/"] = lambda l, r: l / r

# Regular expressions with which string tokens in the Postfix equation are compared.
# String tokens are assumed to be either characters representing arithmetic operations (e.g. "+")
# or integers.
regexOps = re.compile(r"^([\+\-\*\/])$")
regexInts = re.compile(r"^([\d]+)$")

# Create the stack with which to evaluate the Postfix equation.
ps = PostfixStack()

# Iterate through the lines in input.txt.
# Assumption is made that each line represents a Postfix equation.  Postfix equations are 
# evaluated and their final values written to output.txt.
with fileinput.input(files = ("input.txt")) as fin:
    with open("output.txt", "w") as fout:
        for line in fin:
            # Split the line into string tokens
            for token in line.split():
                # Integer? Push it onto the stack.
                m = regexInts.match(token)
                if m is not None:
                    ps.push(int(m.group(1)))
                # Arithmetic operator? Evaluate it with values from the stack, and push the result
                # back upon the stack.
                m = regexOps.match(token)
                if m is not None:
                    r = ps.pop()
                    l = ps.pop()
                    ps.push(ops[m.group(1)](l, r))
            # Write the value of the Postfix equation to the output file.
            fout.write(format(ps.pop(), ".2f") + "\n")

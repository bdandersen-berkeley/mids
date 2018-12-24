import fileinput
import os
import re

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

ops = {}
ops["+"] = lambda l, r: l + r
ops["-"] = lambda l, r: l - r
ops["*"] = lambda l, r: l * r
ops["/"] = lambda l, r: l / r

regexOps = re.compile(r"^([\+\-\*\/])$")
regexInts = re.compile(r"^([\d]+)$")

ps = PostfixStack()

with fileinput.input(files = ("input.txt")) as fin:
    for line in fin:
        for token in line.split():
            m = regexInts.match(token)
            if m is not None:
                ps.push(int(m.group(1)))
            else:
                m = regexOps.match(token)
                if m is not None:
                    l = ps.pop()
                    r = ps.pop()
                    ps.push(ops[m.group(1)](l, r))
                else:
                    print("Invalid syntax")
        print(ps.pop())

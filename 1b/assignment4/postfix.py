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

ps = PostfixStack()
ps.push(51)
ps.push(43)
print(ps.pop())
print(ps.pop())

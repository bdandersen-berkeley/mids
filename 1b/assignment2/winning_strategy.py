def hasWinningStrategy(n, myPlay):

    if (n < 2):
        print("n = " + str(n) + ", returning: " + str(not myPlay))
        return not myPlay

    leftStrategy = hasWinningStrategy(n - 1, not myPlay)
    rightStrategy = hasWinningStrategy(n / 2, not myPlay)
    print("n = " + str(n) + ", left strategy: " + str(leftStrategy) + ", right strategy: " + str(rightStrategy))
    return leftStrategy or rightStrategy

inputIsFloat = False
while False == inputIsFloat:
    firstMove = False
    firstMoveEntry = input("Will Player One make the first move (T/F)? ")
    if str(firstMoveEntry) is "T":
        firstMove = True
    try:
        n = input("Please enter the beginning number: ")
        n = float(n)
        inputIsFloat = True
    except:
        print("The beginning number must be a float - try again")

print("Player One has winning strategy: " + str(hasWinningStrategy(n, firstMove)))
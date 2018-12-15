def playerOneHasWinningStrategy(n, isPlayerOne):

    if (n < 2):
        print("Base condition reached; n: " + str(n) + ", Player one must lose: " + str(isPlayerOne))
        return not isPlayerOne

    print("n: " + str(n) + ", Player one's turn: " + str(isPlayerOne))
    nPrime = n
    return \
        playerOneHasWinningStrategy(n - 1, not isPlayerOne) and \
        playerOneHasWinningStrategy(nPrime / 2, not isPlayerOne)

inputIsFloat = False
while False == inputIsFloat:
    try:
        n = input("Please enter the beginning number: ")
        n = float(n)
        inputIsFloat = True
    except:
        print("The beginning number must be a float - try again")

print("Player One has winning strategy: " + str(playerOneHasWinningStrategy(n, True)))
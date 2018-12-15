import io
import time

def getRunningTime(n):

    # begin = int(round(time.time() * 1000))
    ticks = 0

    i = 0
    j = 1
    ticks += 2

    while i < n:
        ticks += 1
        for k in range(j):
            ticks += 1
            i += 1
            ticks +=1
        j *= 2
        ticks += 1

    # end = int(round(time.time() * 1000)) - begin
    # return end
    return ticks

with open('q3.csv', 'w') as q3:
    for n in range(500):
        q3.write(str(n) + "," + str(getRunningTime(n)) + "\n")
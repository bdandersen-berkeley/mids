import io
import time

def getRunningTime(n):

    begin = int(round(time.time() * 1000000))

    i = n
    j = 1
    while j < 1:
        i *= 100
        j *= 101

    end = int(round(time.time() * 1000000)) - begin
    return end

with open('q2.csv', 'w') as q2:
    for n in range(500):
        q2.write(str(n) + "," + str(getRunningTime(n)) + "\n")
import io
import time

def getRunningTime(n):

    begin = int(round(time.time() * 1000))
    # print("Begin: " + str(begin))

    for i in range(n):
        for j in range(n^2):
            print(i + j)

    end = int(round(time.time() * 1000)) - begin
    # print("End: " + str(end))
    return end

with open('q1.csv', 'w') as q1:
    for n in range(500):
        q1.write(str(n) + "," + str(getRunningTime(n)) + "\n")
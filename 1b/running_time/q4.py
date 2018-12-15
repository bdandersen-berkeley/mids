import io
import time

def getRunningTime(n):

    begin = int(round(time.time() * 1000))

    def func(i):
        if i < 1:
            return 1
        else:
            return func(i - 1) + func(i - 1) 

    print(func(n))

    end = int(round(time.time() * 1000)) - begin
    return end

with open('q4.csv', 'w') as q4:
    for n in range(25):
        q4.write(str(n) + "," + str(getRunningTime(n)) + "\n")
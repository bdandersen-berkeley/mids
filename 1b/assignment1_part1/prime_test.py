def isPrime(n):

    if (n <= 1):
        return False

    max_divisor = n
    cur_divisor = 2

    while (cur_divisor <= max_divisor):

        # print("cur_divisor: {0}, max_divisor: {1}".format(cur_divisor, max_divisor))

        if (cur_divisor == max_divisor):
            return True
        elif (0 == n % cur_divisor):
            # print("{0} is divisible by {1}".format(n, cur_divisor))
            return False
        else:
            max_divisor = n // cur_divisor
            cur_divisor = cur_divisor + 1

    return True

def getLeastPrimeDivisor(n, primes):

    if (n in primes):
        return n

    for prime in primes:
        if (0 == n % prime):
            return prime
        
    return n

primes = set()
for x in range(1, 10000):
    if (isPrime(x)):
        primes.add(x)

n = 1234567890
divisors = list()
while (n not in primes):
    leastPrimeDivisor = getLeastPrimeDivisor(n, sorted(primes))
    divisors.append(leastPrimeDivisor)
    n = n // leastPrimeDivisor
divisors.append(n)

print(divisors)
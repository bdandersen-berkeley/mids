def sumDigits(n):
    sum = 0
    modPair = divmod(n, 10)
    while (modPair[0] != 0):
        sum = sum + modPair[1]
        modPair = divmod(modPair[0], 10)
    return sum + modPair[1]

inputIsInt = False
while False == inputIsInt:
    try:
        phone = input("Please enter your phone number (without punctuation): ")
        phone = int(phone)
        inputIsInt = True
    except:
        print("Your phone number must consist only of digits - try again")

y = phone - sumDigits(phone)
while (y >= 10):
    y = sumDigits(y)
    
print("Answer: ", y)
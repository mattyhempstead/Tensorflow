import math, random

def softmax(arr):
    arr = [math.exp(i) for i in arr]
    s = sum(arr)
    arr = [i/s for i in arr]
    return arr

def normalise(arr):
    s = sum(arr)
    return [i/s for i in arr]


x = [2,1,0.1]

print(softmax(x))

print(normalise(x))


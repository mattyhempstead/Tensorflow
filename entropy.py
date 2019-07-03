import math

def entropy(p):
    return sum([i*math.log(1/i) for i in p])

def bitEntropy(p):
    return sum([i*math.log2(1/i) for i in p])


def crossEntropy(p,q):  
    # The number of bits needed to encode a value from p using the 
    # optimal encoding scheme from q.

    # If used as a loss function, p is the true label and q is the predicted label.
    # Thus p is allowed to have zero's but the predicted value q can not.
    return sum([-p[i] * math.log(q[i]) for i in range(len(p))])

p = [0.1,0.35,0.55]
q = [0.2,0.1,0.7]


print(entropy(p))

print(crossEntropy(q,p))


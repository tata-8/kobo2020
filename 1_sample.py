import numpy as np

def step(x):
    return 1 if x > 0 else 0

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(w*x) + b
    y = step(y)
    return y

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(w*x) + b
    y = step(y)
    return y

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.25
    y = np.sum(w*x) + b
    y = step(y)
    return y

def XOR(x1, x2): 
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    print("AND")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(out))
    
    print("OR")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out = OR(xs[0], xs[1])
        print(str(xs) + " -> " + str(out))
        
    print("NAND")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(out))
        
    print("XOR")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(out))
        
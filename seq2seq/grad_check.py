import numpy as np
from numpy.random import randn
from random import randint

from layers.embedding import Embedding
from layers.lstm import Lstm
from layers.softmax import Softmax

DELTA = 1e-5
THRESHOLD = 1e-2

EOS = 0
HIDDNE_SIZE = 10

input_layers = [
    Embedding(5, 10),
    Lstm(10, 10),
]

output_layers = [
    Embedding(5, 10),
    Lstm(10, 10, previous=input_layers[1]),
    Softmax(10, 4),
]
X = [randint(0, 4), randint(0, 4)]
Y = [randint(0, 3), randint(0, 3)]

def train():
    # reset state
    for layer in input_layers:
        layer.initSequence()
    #forward
    for x in X:
        h = x
        for layer in input_layers:
            h = layer.forward(h)
            
    #reset state
    for layer in output_layers:
        layer.initSequence()
        
    for y in [EOS] + Y:
        h = y
        for layer in output_layers:
            h = layer.initSequence()
            
    #backward
    for y in reversed(Y + [EOS]):
        delta = y
        for layer in reversed(output_layers):
            delta = layer.backward(delta)
            
    for x in reversed(X):
        delta = np.zeros(HIDDNE_SIZE)
        for layer in reversed(input_layers):
            delta = layer.backward(delta)
            
    return output_layers[-1].getCost()

def numeric_gradient(mat, dmat, i):
    val = mat.falt[i]
    
    mat.flat[i] = val + DELTA
    loss1 = train()
    mat.flat[i] = val - DELTA
    loss2 = train()
    
    mat.flat[i] = val
    
    return(loss1 - loss2) / (2 * DELTA)

def main():
    passed = True
    
    for layer in input_layers + output_layers:
        for name, mat, dmat in layer.params:
            for i in xrange(mat.size):
                grad_num = numeric_gradient(mat, dmat, i)
                grad_analystic = dmat.flat[i]
                
                if grad_num == 0 and grad_analystic == 0:
                    continue
                
                error = abs(grad_analystic - grad_num)
                
                if error > THRESHOLD or np.isnan(error):
                    print(layer, name, "ERROR", grad_analystic, grad_num)
                    passed = False
                    break
                
    if passed : 
        print("ALL PASEED")

if __name__ == "__main__":
    main()
import numpy as np 
from tqdm import tqdm

##读取数据
# def load_datasets(fdir):

#     return X, Y, X_test, Y_test

##Relu
class relu:
    def forward(X):
        Y = np.maximum(0,X)
        cache = X
        return Y, cache

    def backward(dY, cache):
        Z = cache
        dZ = np.array(dY, copy=True)
        dZ[Z <= 0] = 0
        return dZ

##sigmoid
class sigmoid:
    def forward(X):
        Y = 1 / (1 + np.exp(-X))
        cache = X
        return Y, cache

    def backward(dY, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dY * s * (1-s)
        return dZ

##初始化参数
def initialize_parameters(layerdims):
    np.random.seed(1)
    parameters = {}
    L = len(layerdims)
    parameters['layerdims']=layerdims;
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layerdims[l-1], layerdims[l])
        parameters['b' + str(l)] = np.random.randn(1, layerdims[l])

    return parameters

##正向传播
def linear_forward(X, W, b):
    Y = np.dot(X, W) + b
    cache  = (X, W, b)
    
    return Y, cache

def activation_forward(X, W, b, activation):
    Z, linear_cache = linear_forward(X,W,b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid.forward(Z)
    elif activation == "relu":
        A, activation_cache = relu.forward(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

def forward(X, parameters):
    A = X
    caches = []
    L = len(parameters['layerdims'])-1
    for l in range(1, L):
        A_l = A
        A, cache = activation_forward(A_l, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
        caches.append(cache)
    
    AL, cache = activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def cost(AL, Y):
    m = Y.shape[0]
    cost = -1 / m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL + 1e-5),axis=0,keepdims=True)
    cost = np.squeeze(cost)
    
    return cost

##反向传播
def linear_backward(dZ, cache):
    A_l, W, b = cache
    m = A_l.shape[0]
    dW = 1 / m * np.dot(A_l.T, dZ)
    db = 1 / m * np.sum(dZ, axis=0, keepdims=True)
    dA_l = np.dot(dZ,W.T) 
    
    return dA_l, dW, db

def activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu.backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid.backward(dA, activation_cache)

    dA_l, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_l, dW, db

def backward(AL, Y, caches):
    grads = {} 
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL + 1e-5) - np.divide(1 - Y, 1-AL + 1e-5))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, "sigmoid")
    # print(grads)
    
    for l in reversed(range(1, L)):
        current_cache = caches[l-1]
        dA_l_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l+1)], current_cache,  "relu")
        grads["dA" + str(l)] = dA_l_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads

##梯度下降
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters['layerdims'])

    for l in range(1, L):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
        
    return parameters

def predict(X, parameters):
    probas, caches = forward(X, parameters)
    p = np.round(probas)

    return p

def train(X, Y, layerdims, learingrate, iterations):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layerdims)
    with tqdm(total=iterations) as t:
        for i in range(0, iterations):
            z, cache = forward(X, parameters)
            costz = np.squeeze(cost(z, Y))
            grads = backward(z, Y, cache)
            parameters = update_parameters(parameters, grads, learingrate)
            costs.append(costz)
            t.set_description('Training %i' % i)
            t.set_postfix(cost=costz, learning_rate=learingrate)
            t.update(1)
    
    return parameters

def save_parameters_to_file(parameters, filename):
    with open(filename, "w") as f:
        f.write("#ifndef __PARAMETER_H__\n")
        f.write("#define __PARAMETER_H__\n\n")
        f.write("#include <stdlib.h>\n\n")

        # Write W matrices
        f.write("double** parameter_w[] = {\n")
        for i, key in enumerate(parameters.keys()):
            if key.startswith('W'):
                f.write("    (double*[]){\n")
                W = parameters[key].T  # Transpose W matrix
                for j in range(W.shape[1]):
                    values = ", ".join(str(value) for value in W[:, j])
                    f.write(f"        (double[]){{{values}, 9999}},\n")
                f.write("        NULL\n    },\n")
        f.write("    NULL\n};\n\n")

        # Write b matrices
        f.write("double** parameter_b[] = {\n")
        for i, key in enumerate(parameters.keys()):
            if key.startswith('b'):
                f.write("    (double*[]){\n")
                b = parameters[key].T  # Transpose b matrix
                values = ", ".join(str(value) for value in b.flatten())
                f.write(f"        (double[]){{{values}, 9999}},\n")
                f.write("        NULL\n    },\n")
        f.write("    NULL\n};\n\n")

        f.write("#endif\n")
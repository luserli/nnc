import numpy as np
from utils import *
X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y=np.array([[0], [1], [1], [0]])
layerdims = [X.shape[1], 3, 2, 1]
parameters = train(X, Y, layerdims, 0.05, 5000)
save_parameters_to_file(parameters, '../Inc/parameter.h')
Y_predict = predict(X, parameters)
print("训练集准确性："  , format(100 - np.mean(np.abs(Y_predict - Y)) * 100) ,"%")
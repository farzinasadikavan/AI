import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# labels = pd.read_csv('labels.csv')
# dataByArray=[]
# for index , row in labels.iterrows():
#     img=Image.open('data/'+str(row['id'])+'.png').convert('L')
#     arr=np.asarray(img)
#     dataByArray.append([arr,row['id'],row['label']])
# data=pd.DataFrame(dataByArray)
# data.rename({0: 'img', 1: 'id',2: 'label'}, axis=1,inplace=True)
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(data['label'])
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# onehot_encoded=np.matrix(onehot_encoded)
# print(type(onehot_encoded))
class Tanh:
    
    def __init__(self): pass

    def __val(self, matrix):
        '''
        This private method gets a matrix and uses the activity function on that.
        It performs Tanh on the values.
        
            Parameters:
                matrix: np.matrix of values
            Returns:
                tanh_value: np.matrix of Tanh activation function result
        '''
        tanh_value=np.tanh(matrix)
        return tanh_value

    def derivative(self, matrix):
        '''
        Returns the derivation value of Tanh function on input matrix.
        
            Parameters:
                matrix: np.matrix of values
            Returns:
                sigmoid_derivative: np.matrix of Tanh activation function derivation result
        '''
        arr=np.asarray(matrix)
        arr=1.0 - np.tanh(arr)**2
        tanh_derivative=np.asmatrix(arr)
        return tanh_derivative
    
    def __call__(self, matrix):
        '''
        __call__ is a special function in Python that, when implemented inside a class,
        gives its instances (objects) the ability to behave like a function.
        Here we return the _val method output.
            
            Parameters:
                matrix: np.matrix of values
            Returns:
                __val(matrix): __val return value for the input matrix
        '''
        return self.__val(matrix)


sigmoid=Tanh()
print(sigmoid.__call__(np.asmatrix([[1, 0, 0.2, 3],[1,  -1,   7, 3],[2,  12,  13, 3]])))
print("derivative")
print(sigmoid.derivative(np.asmatrix([[0, 0.4,2,4], [0.5, 0.7,7,6], [0.9, .004,8,5]])))

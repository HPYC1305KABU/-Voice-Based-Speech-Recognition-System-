import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy as sp 
import sklearn
import random 
import time 
from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing, model_selection


from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.models import model_from_json

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization


data = pd.read_csv('Main_file.csv')
data = shuffle(data)

#for test data------------------------------------------
data_test = pd.read_csv('Test.csv')
data_test = shuffle(data_test)
#--------------------------------------------

i = 3
#data_to_predict = data_test[:i].reset_index(drop = True)
#predict_name = data_to_predict.name_id 
#predict_name = np.array(predict_name)
#prediction = np.array(data_to_predict.drop(['name'],axis= 1))

data = data.reset_index(drop = True)


X = data.drop(['name'], axis = 1)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

X = np.array(X)
Y = data['name']

# Transform name species into numerical values 
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)
#print(Y)

#for test data-----------------------------------------------------
X_test = data_test.drop(['name'], axis = 1)

scaler = MinMaxScaler(feature_range=(0, 1))
X_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)

X_test = np.array(X_test)
#Y_test = data_test['name']

# Transform name species into numerical values 
#encoder = LabelEncoder()
#encoder.fit(Y_test)
#Y_test = encoder.transform(Y_test)
#Y_test = np_utils.to_categorical(Y_test)
#print(Y)
#------------------------------------------------------

input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(100,input_dim = input_dim , activation = 'tanh'))
#model.add(Dense(135, activation = 'relu'))
#model.add(Dense(100, activation = 'relu'))
#model.add(Dense(80, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(70))
model.add(LeakyReLU(alpha=[0.05]))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(LeakyReLU(alpha=[0.05]))
model.add(BatchNormalization())
model.add(Dense(9, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(X, Y, epochs = 20, batch_size = 15)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))






# serialize model to JSON
model_json = model.to_json()
with open("xyz.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("xyz.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('xyz.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("xyz.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))






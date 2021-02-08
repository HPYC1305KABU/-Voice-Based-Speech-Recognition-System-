import numpy as np 
import pandas as pd 
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#import scipy as sp 
#import sklearn
#import random 
#import time 
from sklearn.preprocessing import MinMaxScaler

#from sklearn import preprocessing, model_selection


from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.models import model_from_json

data = pd.read_csv('Main_file.csv')
data = shuffle(data)

#for test data------------------------------------------
data_test = pd.read_csv('Test_Data.csv')
data_test = shuffle(data_test)

#--------------------------------------------

i = 200
#data_to_predict = data_test[:i].reset_index(drop = True)
#predict_name = data_to_predict.name 
#predict_name = np.array(predict_name)
#prediction = np.array(data_to_predict.drop(['name'],axis= 1))

#data = data.reset_index(drop = True)


#X = data.drop(['name'], axis = 1)

#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)
#X = pd.DataFrame(X)

#X = np.array(X)
Y = data['name']

# Transform name species into numerical values 
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)
#print(Y)

#for test data-----------------------------------------------------
X_test = data_test.drop(['name'], axis = 1)
#X_test = data.dropna(axis = 0, how ='any') 

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

# later...
 
# load json and create model
json_file = open('xyz.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("xyz.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


for j in range(i):

	print('-------------------------------------------------------')

	predictions = loaded_model.predict_classes(X_test)
	#prediction_ = np.argmax(to_categorical(predictions), axis=1)
	#prediction_ = np.argsort(predictions, axis=-1, kind='quicksort', order=None)
	prediction_ = np.argsort(to_categorical(predictions[j]))[:-9:-1]

	prediction_ = encoder.inverse_transform(prediction_)
	#print(prediction_)
	##	print( " the nn predict {}, and the brand to find is {}".format(i,j))

	print("----------------------------------------------------------------------------------------------")


	pred = loaded_model.predict_proba(X_test)


	dfe = pred[j]*100 
	wer = np.sort(pred[j]*100)[:-9:-1]


	abc = dict(zip(prediction_,wer))
	print(abc)
		#print(wer)
	

# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from numpy import genfromtxt


# Import old faithful temperature/time data
# Dataset stretches from 4/13/2015 - 10/26/2015
# Temperature in degrees C
# Time measured once per minute. Time in Unix seconds
fname = '/Users/ckruse/Documents/python/OldFaithful/Old_Faithful_Logger.csv'
temp = genfromtxt(fname, delimiter = ',')
time = temp[1:-1,0]
temp = temp[1:-1,1]

# Normalize temperatures between -1 and 1
temp = temp/np.max(temp)*2 - 1
np.shape(temp)
print("Max Temp:", np.max(temp))
print("Min Temp:", np.min(temp))
plt.hist(temp,100)
plt.title("Histogram of normalized temperatures")
plt.show()

# Plot a few snippets of data of length 2000
for i in range(1,4):
	plot_length = 2000
	random_index = np.random.randint(0, len(temp) - plot_length)
	plt.plot(temp[random_index:random_index + plot_length], linewidth = 0.5)
	plt.title("Start Index: " + str(random_index))
	plt.show()



def create_dataset(dataset, labels, look_back):
	dataX, dataY = [], []
	for i in range(0,len(dataset)-look_back-1,1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(labels[i + look_back])
	return np.array(dataX), np.array(dataY)

timing = []
i = 0
start = 0
for measurement in range(0, len(temp)):

    if temp[measurement] < -0.10:
        timing.append(i)
        i += 1

    else:
        i = 0
        timing.append(i)
        end = measurement
        timing[start:end] = np.max(timing[start:end])-timing[start:end]
        start = measurement

np.argmax(timing)
print(np.max(timing))

timing = np.divide(timing,np.max(timing))*2 - 1
print(np.max(timing))
print(np.max(temp))


val = 0.1
seq_len = 100

X, Y = create_dataset(temp[0:int(len(temp)-val*len(temp))], timing[0:int(len(timing)-val*len(timing))], seq_len)
X_val, Y_val = create_dataset(temp[int(len(temp)-val*len(temp)):-1], timing[int(len(timing)-val*len(timing)):-1], seq_len)

temp_val = temp[int(len(temp)-val*len(temp)):-1]


print(X.shape)
X = X.reshape(X.shape[0], X.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
Y = Y.reshape(Y.shape[0])

print(X.shape)
print(Y.shape)


from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten

model = Sequential()

model.add(Conv1D(filters=4, kernel_size=4, activation='relu', input_shape=(np.shape(X)[1],np.shape(X)[2]), strides=4))
model.add(Conv1D(filters=4, kernel_size=4, activation='relu', input_shape=(np.shape(X)[1],np.shape(X)[2]), strides=4))
#model.add(Conv1D(filters=8, kernel_size=16, activation='relu', strides=4))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))


model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.sgd(),
              metrics=['mse'])

from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, LSTM, AveragePooling1D

# model = Sequential()
# #model.add(Conv1D(filters=8, kernel_size=4, activation='relu', input_shape=(np.shape(X)[1],np.shape(X)[2]), strides=3))
# #model.add(AveragePooling1D(pool_size=2))
# #model.add(Conv1D(filters=8, kernel_size=4, activation='relu', strides=3))
# #model.add(Flatten())
# model.add(LSTM(16, return_sequences=True, input_shape=(np.shape(X)[1],np.shape(X)[2]), activation='tanh'))
# model.add(LSTM(16, activation='tanh'))
# #model.add(Dense(8, activation='relu'))
# #model.add(Dropout(0.5))
# #model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear'))


model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.sgd(),
              metrics=['mse'])

from keras.models import load_model

epochs = 150

#model = load_model('/Users/ckruse/Documents/python/oldFaithfulTemp.h5')
history = model.fit(X,
                  Y,
                  epochs=epochs,
                  validation_data = [X_val,Y_val],
                  verbose=1,
                  batch_size=256,
                  shuffle=True)

model.save('/Users/ckruse/Documents/python/oldFaithfulTemp.h5')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
plt.show()

#score = model.evaluate(data_stack, label_stack, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#model.save('/Users/ckruse/Documents/python/soccer/model.h5')


scope = range(0,20000)
plot_range = range(0,np.max(scope)-np.min(scope)+1)

testPredict = model.predict(X_val[scope])
testPredict = ((testPredict+1)/2)*196

trainPredict = model.predict(X[0:150])
trainPredict = ((trainPredict+1)/2)*196
normY = ((Y_val+1)/2)*196


plt.figure(figsize=(15,10))
plt.plot(testPredict[plot_range,0])
plt.scatter(plot_range,normY[scope],s=3, c=[1,0,0])
plt.xlabel('Time Step (min)')
plt.ylabel('Predicted time until next eruption (min)')
plt.title('Time Until Next Old Faithful Eruption')
plt.legend(['Predicted', 'Actual'])
plt.show()

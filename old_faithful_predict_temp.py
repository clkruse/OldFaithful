# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from numpy import genfromtxt


# Import old faithful temperature/time data
# Dataset stretches from 4/13/2015 - 10/26/2015
# Temperature in degrees C
# Time measured once per minute. Time in Unix seconds
fname = '/Users/ckruse/Documents/python/OldFaithful/Old_Faithful_Logger.csv'
data = genfromtxt(fname, delimiter = ',')
time = data[1:-1,0]
temp = data[1:-1,1]

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


# Cut the last n percent of the dataset off as a test set
test_percentage = 1 - 0.1

temp_train = temp[0:int(test_percentage * len(temp))]
temp_test = temp[int(test_percentage * len(temp)) : len(temp)]


# Select random snippets of consistent sequence lengths from the train dataset
# First portion is input, second portion is labels

def select_sequences(data, input_length, output_length, num_sequences):

    x = np.zeros([num_sequences, input_length])
    y = np.zeros([num_sequences, output_length])

    for index in range(num_sequences):
        input_start = np.random.randint(0,len(temp) - input_length * 2)
        input_end = input_start + input_length

        output_start = input_start + input_length
        output_end = input_start + input_length + output_length

        x[index,:] = temp[range(input_start, input_end)]
        y[index,:] = temp[range(output_start, output_end)]

    t = range(0, input_length + output_length)
    plt.plot(t[0 : input_length], x[0])
    plt.plot(t[input_length : input_length + output_length], y[0])
    plt.legend(["Input", "Output"])
    plt.title("Single sequence of input and output data")
    plt.show()
    return [x, y]


input_length = 300
output_length = 60
num_train_sequences = 500000
num_test_sequences = 5000

[x_train, y_train] = select_sequences(temp_train, input_length, output_length, num_train_sequences)

[x_test, y_test] = select_sequences(temp_test, input_length, output_length, num_test_sequences)

# Reshape data to make keras happy??

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
print("input train data shape:", x_train.shape, "\noutput train data shape:", y_train.shape)
print("input test data shape:", x_test.shape, "\noutput test data shape:", y_test.shape)



model = Sequential()

model.add(Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(np.shape(x_train)[1],np.shape(x_train)[2]), strides=2))
#model.add(Conv1D(filters=8, kernel_size=4, activation='relu', input_shape=(np.shape(x_train)[1],np.shape(x_train)[2]), strides=4))
model.add(Conv1D(filters=32, kernel_size=4, activation='relu', strides=2))



#model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train.shape[1], activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.adam(),
              metrics=['mae'])




epochs = 75

#model = load_model('/Users/ckruse/Documents/python/old_faithful_sequence_preds.h5')
history = model.fit(x_train,
                  y_train,
                  epochs=epochs,
                  validation_data = [x_test,y_test],
                  verbose=1,
                  batch_size=32,
                  shuffle=True)

#model.save('/Users/ckruse/Documents/python/old_faithful_sequence_preds.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


predict_test = model.predict(x_test[10:20])
for i in range(0,10):
    t = range(0, input_length + output_length)
    plt.plot(t[0 : input_length], x_test[i])
    plt.plot(t[input_length : input_length + output_length], predict_test[i])
    plt.plot(t[input_length : input_length + output_length], y_test[i])
    plt.legend(["Inputs", "Predictions", "Labels"])
    plt.show()

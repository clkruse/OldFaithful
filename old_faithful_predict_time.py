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

# Create a vector of delta temperatures
delta = temp[1:len(temp)] - temp[0:len(temp)-1]
delta = np.insert(delta, 0, 0)

threshold = 0.2
eruptions = delta > threshold

time_to_eruption = []
counter = 0
for value in range(len(eruptions)-1,-1,-1):
	time_to_eruption.append(counter)
	counter += 1
	if eruptions[value] == True:
		counter = 0

# Normalize time to eruption between 0 and 1
scale_factor = np.max(time_to_eruption)
time_to_eruption = time_to_eruption/np.max(time_to_eruption)
time_to_eruption = time_to_eruption[::-1]

plt.plot(time_to_eruption[0:1000])
plt.plot(temp[0:1000])
plt.show()

# Cut the last n percent of the dataset off as a test set
test_percentage = 1 - 0.1

[time_train, temp_train] = [time_to_eruption[0:int(test_percentage * len(time_to_eruption))], temp[0:int(test_percentage * len(temp))]]

[time_test, temp_test] = [time_to_eruption[int(test_percentage * len(time_to_eruption)) : len(time_to_eruption)], temp[int(test_percentage * len(temp)) : len(temp)]]

# Select random snippets of consistent sequence lengths from the train dataset
# First portion is input, second portion is labels

def select_sequences(input_data, labels, input_length, num_sequences):

    x = np.zeros([num_sequences, input_length])
    y = np.zeros(num_sequences)

    for index in range(num_sequences):
        input_start = np.random.randint(0,len(input_data) - input_length * 2)
        input_end = input_start + input_length
        x[index,:] = input_data[range(input_start, input_end)]
        y[index] = labels[input_end]
    plt.plot(x[0])
    plt.scatter(input_length, y[0],c='r')
    plt.grid()
    plt.show()
    return [x, y]


input_length = 300
num_train_sequences = 100000
num_test_sequences = 10000

[x_train, y_train] = select_sequences(temp_train, time_train, input_length, num_train_sequences)

[x_test, y_test] = select_sequences(temp_test, time_test, input_length, num_test_sequences)

# Reshape data to make keras happy??

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
print("input train data shape:", x_train.shape, "\noutput train data shape:", y_train.shape)
print("input test data shape:", x_test.shape, "\noutput test data shape:", y_test.shape)



model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.adam(),
              metrics=['mae'])




epochs = 25

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

evaluation_len = 500
random_index = np.random.randint(0, len(time_test) - evaluation_len)
model_validation = np.zeros([evaluation_len,input_length,1])
model_validation.shape
preds = []

for i in range(evaluation_len):
	model_validation[i,:,0] = temp_train[i+random_index:input_length + random_index + i]

preds = model.predict(model_validation) * scale_factor
labels = time_test[0:evaluation_len] * scale_factor
plt.plot(preds)
plt.plot(labels)
plt.show()






predict_test = model.predict(x_test[10:20])
for i in range(0,10):
    t = range(0, input_length + output_length)
    plt.plot(t[0 : input_length], x_test[i])
    plt.plot(t[input_length : input_length + output_length], predict_test[i])
    plt.plot(t[input_length : input_length + output_length], y_test[i])
    plt.legend(["Inputs", "Predictions", "Labels"])
    plt.show()

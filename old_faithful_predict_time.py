# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import keras
from keras.models import load_model
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

threshold = 0.1
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

for i in range(2):
	plot_length = 1000
	random_index = np.random.randint(0, len(time_to_eruption) - plot_length)
	plt.plot(time_to_eruption[random_index:random_index + plot_length], linewidth = 0.5)
	plt.plot(temp[random_index:random_index + plot_length], linewidth = 0.5)
	plt.title("Start Index: " + str(random_index))
	plt.grid()
	plt.show()



# Cut the last n percent of the dataset off as a test set
test_percentage = 1 - 0.1

[time_train, temp_train] = [time_to_eruption[0:int(test_percentage * len(time_to_eruption))], temp[0:int(test_percentage * len(temp))]]

[time_test, temp_test] = [time_to_eruption[int(test_percentage * len(time_to_eruption)) : len(time_to_eruption)], temp[int(test_percentage * len(temp)) : len(temp)]]

# Select random snippets of consistent sequence lengths from the train dataset
# First portion is input, second portion is labels

def select_random_sequences(input_data, labels, input_length, num_sequences):

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


#input_length = 150
#num_train_sequences = 100000
#num_test_sequences = 10000

#[x_train, y_train] = select_random_sequences(temp_train, time_train, input_length, num_train_sequences)

#[x_test, y_test] = select_random_sequences(temp_test, time_test, input_length, num_test_sequences)


def select_sequential_sequences(input_data, labels, input_length):

    x = np.zeros([len(input_data)-input_length, input_length])
    y = labels[input_length - 1:-1]

    for i in range(len(input_data) - input_length):
        x[i, :] = input_data[i:i+input_length]
    plt.plot(x[0])
    plt.scatter(input_length, y[0],c='r')
    plt.grid()
    plt.show()
    return [x, y]

input_length = 100

[x_train, y_train] = select_sequential_sequences(temp_train, time_train, input_length)

[x_test, y_test] = select_sequential_sequences(temp_test, time_test, input_length)


# Reshape data to make keras happy??

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
print("input train data shape:", x_train.shape, "\noutput train data shape:", y_train.shape)
print("input test data shape:", x_test.shape, "\noutput test data shape:", y_test.shape)


def train_and_evaluate(x_train, y_train, x_test, y_test, epochs, num_convs, conv_size, dense_width, dense_depth):

	model = Sequential()
	model.add(Conv1D(filters=num_convs, kernel_size=conv_size, activation='relu', input_shape=(np.shape(x_train)[1],np.shape(x_train)[2]), strides=2))
	#model.add(Conv1D(filters=8, kernel_size=4, activation='relu', input_shape=(np.shape(x_train)[1],np.shape(x_train)[2]), strides=4))
	#model.add(Conv1D(filters=16, kernel_size=6, activation='relu', strides=2))
	#model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
	#model.add(Dropout(0.5))

	model.add(Flatten())

	for i in range(dense_depth):
		model.add(Dense(dense_width, activation='relu'))

	model.add(Dense(1, activation='linear'))
	model.compile(loss=keras.losses.mean_squared_error,
	              optimizer=keras.optimizers.adam(),
	              metrics=['mae'])

	#model = load_model('old_faithful_time.h5')
	history = model.fit(x_train,
	                  y_train,
	                  epochs=epochs,
	                  validation_data = [x_test,y_test],
	                  verbose=0,
	                  batch_size=32,
	                  shuffle=True)

	model.save('old_faithful_time.h5')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.show()

	return([model, history.history, history.params])

for i in range(0, 10):
	epochs = 20
	num_convs = 8
	conv_size = 3
	dense_width = 32
	dense_depth = i

	[model, history, params] = train_and_evaluate(x_train, y_train, x_test, y_test, epochs, num_convs, conv_size, dense_width, dense_depth)
	visualize_network(temp_test, time_test, input_length, scale_factor, model, "dense depth: " + str(dense_depth))


for i in range(1, 10):
	epochs = 20
	num_convs = 8
	conv_size = i * 2
	dense_width = 32
	dense_depth = 3

	[model, history, params] = train_and_evaluate(x_train, y_train, x_test, y_test, epochs, num_convs, conv_size, dense_width, dense_depth)
	visualize_network(temp_test, time_test, input_length, scale_factor, model, "Conv Size: " + str(conv_size))





visualize_network(temp_train, time_train, input_length, scale_factor, model, "dense depth: " + str(dense_depth))





visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length short
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 600
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, double convs (8, 16)
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, more convs (8, 16), more fc (6x32)
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, no convs, 3x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, no convs, 1x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, no convs, 2x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, 8 c6, 2x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, 8 c16, 2x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, 8 c3, 2x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, 8 c3, 10x32 fc
visualize_network(temp_test, time_test, input_length, scale_factor, model)


#sequence length 100, 8 c3, 2x32 fc, batch size 32 (rather than 256)
visualize_network(temp_test, time_test, input_length, scale_factor, model)

#sequence length 100, 8 c3, 2x32 fc, batch size 32, mae rather than mse
visualize_network(temp_test, time_test, input_length, scale_factor, model)

evaluation_len = 500
random_index = np.random.randint(0, len(time_test) - evaluation_len)
model_validation = np.zeros([evaluation_len,input_length,1])
model_validation.shape
preds = []

for i in range(evaluation_len):
	model_validation[i,:,0] = temp_test[i+random_index:input_length + random_index + i]

preds = model.predict(model_validation) * scale_factor
labels = time_test[random_index+ input_length:random_index + evaluation_len + input_length] * scale_factor
plt.plot(preds, linewidth = 0.8)
plt.plot(labels, c = 'r', linewidth = 0.8)
plt.grid()
plt.legend(["Predictions", "Labels"])
plt.show()


def visualize_network(data, labels, input_length, scale_factor, model, title):

	sequential_test = np.zeros([len(data)-input_length, input_length, 1])
	for i in range(len(data) - input_length):
		sequential_test[i, :, 0] = data[i:i+input_length]

	test_preds = np.squeeze(model.predict(sequential_test)) * scale_factor
	test_labels = labels[input_length - 1:-1] * scale_factor

	delta = np.squeeze(np.array([test_preds - test_labels]))
	preds_mean = []
	preds_std = []
	for time in range(1, int(max(test_labels))):
		preds_mean.append(np.mean(delta[test_labels == time]))
		preds_std.append(np.std(delta[test_labels == time]))

	plt.errorbar(range(len(preds_mean)), preds_mean, np.sqrt(preds_std), elinewidth = 0.5, capsize=0, c='r', ecolor = '#1f77b4')
	plt.plot(range(int(min(test_labels)), int(max(test_labels))), np.zeros(int(max(test_labels))), c='gray', linewidth=0.6)
	plt.xlabel("Time to Eruption (min)")
	plt.ylabel("Prediction Error (min)")
	plt.xlim([95, 0])
	plt.ylim([-15, 15])
	plt.title("Prediction Mean/Variance vs. Time to Eruption: " + title)
	plt.show()

	#plt.hist(delta, 1000)
	#plt.xlabel("Predicted/Actual Delta (min)")
	#plt.xlim([-50, 50])
	#plt.show()


random_index = np.random.randint(0, len(time_test) - evaluation_len)
for i in range(250):
	plt.plot(temp_test[i + random_index:input_length + random_index + i] * scale_factor)
	plt.scatter(input_length, time_test[i + random_index + input_length]* scale_factor,c='r')
	#plt.scatter(input_length, preds[i],c='g')
	plt.ylim([-120, 120])
	plt.grid()
	plt.show()



epochs = 20
num_convs = 8
conv_size = 3
dense_width = 10
dense_depth = 1

[model, history, params] = train_and_evaluate(x_train, y_train, x_test, y_test, epochs, num_convs, conv_size, dense_width, dense_depth)
visualize_network(temp_test, time_test, input_length, scale_factor, model, "Conv Size: " + str(conv_size))

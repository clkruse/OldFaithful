import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import os
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D
from keras import backend as K
from keras.utils import to_categorical

[time, temp] = import_data('Old_Faithful_Logger.csv')

#animated_plot(temp, 1000, 65, 500, 30)

time_to_eruption = create_erupt_countdown(temp, 2.0)

[temp_norm, temp_scale] = normalize_data(temp)

time = to_categorical(time_to_eruption)

#[time_norm, time_scale] = normalize_data(time_to_eruption)

[temp_train, time_train, temp_test, time_test] = split_train_test(temp_norm, time, 0.1)

[x_train, y_train] = select_sequential_sequences(temp_train, time_train, 75)
[x_test, y_test] = select_sequential_sequences(temp_test, time_test, 75)



#[x_train, y_train] = filter_outliers(x_train, y_train, 95)
#[x_test, y_test] = filter_outliers(x_test, y_test, 95)

model = define_model(x_train, y_train)

[model, history, params] = train_model(model, 10, x_train, y_train, x_test, y_test)


# Normalize Data
def normalize_data_broken(data, low, high):
	min = np.min(data)
	max = np.max(data)
	norm_data = (high + low) * (data - min) / (max - min) + low
	scale_factor = (low * max - high * min) / (max - min)
	# Tests
	if low >= high:
		sys.exit("Failed normalization! Low value is greater than or equal to high value")
	if np.max(norm_data) != high:
		sys.exit("Failed normalization! - High val check")
	if np.min(norm_data) != low:
		sys.exit("Failed normalization! - Low val check")

	return norm_data, scale_factor


# Normalize Data Simplistic
def normalize_data(data):
	min = np.min(data)
	max = np.max(data)
	norm_data = data / max
	scale_factor = max
	return norm_data, scale_factor


# Import Data
def import_data(fname):
	# Import old faithful temperature/time data
	# Dataset stretches from 4/13/2015 - 10/26/2015
	# Temperature in degrees C
	# Time measured once per minute. Time in Unix seconds
	data = np.genfromtxt(fname, delimiter = ',')
	time = data[1:-1,0]
	temp = data[1:-1,1]

	return time, temp


# Create the time to eruption vector. For each temperature, there is an associated time until the next eruption
def create_erupt_countdown(data, threshold_std):
	# Create a vector of delta temperatures
	delta = data[1:len(data)] - data[0:len(data)-1]
	delta = np.insert(delta, 0, 0)
	threshold = threshold_std * np.std(delta) + np.mean(delta)
	eruptions = delta > threshold
	print(sum(eruptions), "eruptions are detected at a threshold of", threshold_std)

	erupt_countdown = []
	counter = 0
	temp_std = np.std(temp)
	temp_mean = np.mean(temp)
	for value in range(len(eruptions)-1,-1,-1):
		if temp[value] >=  temp_std * 0.8 + temp_mean:
			erupt_countdown.append(0)

		else:
			erupt_countdown.append(counter)
		counter += 1

		if eruptions[value] == True:
			counter = 0

	erupt_countdown = erupt_countdown[::-1]
	return erupt_countdown


# Split the train and test datasets
def split_train_test(data, labels, percentage):
	train_percentage = 1 - percentage

	[train_data, train_labels] = [data[0:int(train_percentage * len(data))], labels[0:int(train_percentage * len(labels))]]

	[test_data, test_labels] = [data[int(train_percentage * len(data)) : len(data)], labels[int(train_percentage * len(labels)) : len(labels)]]

	print("Number of training samples:", len(train_data))
	print("Number of test samples:", len(test_data))

	return [train_data, train_labels, test_data, test_labels]


# Select sequences with a sliding window across the dataset
def select_sequential_sequences(input_data, input_labels, sequence_len):

    output_data = np.zeros([len(input_data)-sequence_len, sequence_len])
    output_labels = input_labels[sequence_len - 1:-1]

    for i in range(len(input_data) - sequence_len):
        output_data[i, :] = input_data[i:i+sequence_len]

	# Reshape data to make keras happy??
    output_data = output_data.reshape(output_data.shape[0],
    output_data.shape[1], 1)

    return [output_data, output_labels]


# Select random snippets of consistent sequence lengths from the train dataset
def select_random_sequences(input_data, input_labels, sequence_len, num_sequences):

    output_data = np.zeros([num_sequences, sequence_len])
    output_labels = np.zeros(num_sequences)

    for index in range(num_sequences):
        input_start = np.random.randint(0,len(input_data) - sequence_len * 2)
        input_end = input_start + sequence_len
        output_data[index,:] = input_data[range(input_start, input_end)]
        output_labels[index] = input_labels[input_end]

    return [output_data, output_labels]

def filter_outliers(x_data, y_data, time_threshold):
	threshold = np.array(y_data < time_threshold)
	print("Number of samples removed", len(y_data) - sum(threshold))
	x_filtered = x_data[threshold, :, :]
	y_filtered = y_data[threshold]

	return x_filtered, y_filtered


def clipped_msle(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), 0.5) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), 0.5) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

def clipped_mse(y_true, y_pred):
    return K.mean(K.clip(K.square(y_pred - y_true), -0.5, 0.5), axis=-1)

def define_model(input_x, input_y):

	model = Sequential()
	model.add(Conv1D(filters=8, kernel_size=8, activation='relu', input_shape=(np.shape(input_x)[1],np.shape(input_x)[2]), strides=4))
	model.add(MaxPooling1D(pool_size=2))
	#model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.adam(),
				  metrics=['categorical_accuracy'])
	return model

def train_model(model, epochs, x_train, y_train, x_test, y_test):
	#model = load_model('old_faithful_time.h5')
	history = model.fit(x_train,
	                  y_train,
	                  epochs=epochs,
	                  validation_data = [x_test,y_test],
	                  verbose=1,
	                  batch_size=32,
	                  shuffle=True)

	#model.save('old_faithful_time.h5')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'], c='r')
	plt.legend(["Train Loss", "Test Loss"])
	plt.show()

	return([model, history.history, history.params])


def visualize_network(model, data, labels, sequence_len, scale_factor_x, scale_factor_y, title):

	sequential_test = np.zeros([len(data)-sequence_len, sequence_len, 1])

	for i in range(len(data) - sequence_len):
		sequential_test[i, :, 0] = np.squeeze(data[i:i+sequence_len,0])

	test_preds = np.rint(np.squeeze(model.predict(sequential_test)) * scale_factor_y)
	test_labels = np.rint(np.squeeze(labels[0:-1 - sequence_len + 1] * scale_factor_y))

	delta = np.squeeze(np.array([test_preds - test_labels]))

	preds_mean = []
	preds_std = []

	for time in range(1, int(np.max(test_labels))):
		preds_mean.append(np.mean(delta[test_labels == time]))
		preds_std.append(np.std(delta[test_labels == time]))


	plt.errorbar(range(len(preds_mean)), preds_mean, np.sqrt(preds_std), elinewidth = 0.5, capsize=0, c='r', ecolor = '#1f77b4')
	plt.plot(range(int(np.min(test_labels)), int(np.max(test_labels))), np.zeros(int(np.max(test_labels))), c='gray', linewidth=0.6)
	plt.xlabel("Time to Eruption (min)")
	plt.ylabel("Prediction Error (min)")
	#plt.xlim([95, 0])
	#plt.ylim([-15, 15])
	plt.title("Prediction Mean/Variance vs. Time to Eruption: " + str(title))
	plt.show()


def animated_plot(data, labels, max_x, max_y, frames, interval):

	fig = plt.figure()
	ax = plt.axes(xlim = (0,max_x), ylim = (0, max_y))
	line, = ax.plot([], [])

	def init():
		line.set_data([], [])
		return line,

	def animate(i):
		line.set_data(range(max_x), data[i:i+max_x])
		ax.set_title("Time to Eruption: " + str(labels[i+max_x]))
		return line,

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
	HTML(anim.to_html5_video())

	anim._repr_html_() is None
	rc('animation', html='html5')
	return anim


def prediction_overlay(x_test, y_test, model, evaluation_len):

	preds = model.predict(x_test)

	deltas = np.zeros(len(preds))
	pred_max = np.zeros(len(preds))
	label_max = np.zeros(len(preds))

	for i in range(len(preds)):
		pred_max[i] = np.argmax(preds[i])
		label_max[i] = np.argmax(y_test[i])
		diff = pred_max[i] - label_max[i]
		deltas[i] = diff

	random_index = np.random.randint(0, len(x_test) - evaluation_len)

	plt.plot(label_max[random_index:random_index + evaluation_len])
	plt.plot(deltas[random_index:random_index + evaluation_len], c='grey')
	plt.plot(pred_max[random_index:random_index + evaluation_len], c='r')
	plt.legend(["Labels", "Predictions", "Delta"])
	plt.grid()
	plt.show()

prediction_overlay(x_test, y_test, model, 500)

for i in range(1500, 1600):
	plt.plot(preds[i], linewidth=0.5)
	plt.plot(y_test[i]*np.max(preds[i]), c='black')
	plt.ylim([0, np.max(preds[i]) + 0.1 * np.max(preds[i])])
	plt.grid()
	plt.show()


deltas = np.zeros(len(preds))
pred_max = np.zeros(len(preds))
label_max = np.zeros(len(preds))

for i in range(len(preds)):
	pred_max[i] = np.argmax(preds[i])
	label_max[i] = np.argmax(y_test[i])
	diff = pred_max[i] - label_max[i]
	deltas[i] = diff



random_index = np.random.randint(0, len(predictions) - evaluation_len)

plt.plot(label_max[0:500])
plt.plot(deltas[0:500], c='grey')
plt.plot(pred_max[0:500], c='r')
plt.legend(["Labels", "Predictions", "Delta"])
plt.grid()
plt.show()

np.argmax(y_test[0:30])

plt.plot(preds[30])

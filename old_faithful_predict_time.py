import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import os
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling2D, UpSampling2D
from keras import backend as K


[time, temp] = import_data('Old_Faithful_Logger.csv')

#animated_plot(temp, 1000, 65, 200, 30)

time_to_eruption = create_erupt_countdown(temp, 2.0)

[temp_norm, temp_scale] = normalize_data(temp)

[time_norm, time_scale] = normalize_data(time_to_eruption)

#animated_plot(time_norm, 1000, 1, 200, 30)

[temp_train, time_train, temp_test, time_test] = split_train_test(temp_norm, time_norm, 0.1)

for i in range(4):
	plot_length = 1000
	random_index = np.random.randint(0, len(temp_train) - plot_length)
	plt.plot(temp_test[random_index:random_index + plot_length], linewidth = 1)
	plt.plot(time_test[random_index:random_index + plot_length], linewidth = 1)
	plt.title("Start Index: " + str(random_index))
	plt.show()

[x_train, y_train] = select_sequential_sequences(temp_train, time_train, 150)

[x_test, y_test] = select_sequential_sequences(temp_test, time_test, 150)



model = define_model(x_train, y_train)

[model, history, params] = train_model(model, 15, x_train, y_train, x_test, y_test)

visualize_network(model, x_test, y_test, 150, temp_scale, time_scale, "Test")


np.sqrt(0.0035) * time_scale

plt.plot(temp_train[0:1000])
plt.plot(time_train[0:1000], c='r')
plt.show()
plt.plot(temp_test[0:1000])
plt.plot(time_test[0:1000], c='r')
plt.show()


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
def select_sequential_sequences(input_data, input_labels, sequence_length):

    output_data = np.zeros([len(input_data)-sequence_length, sequence_length])
    output_labels = input_labels[sequence_length - 1:-1]

    for i in range(len(input_data) - sequence_length):
        output_data[i, :] = input_data[i:i+sequence_length]

	# Reshape data to make keras happy??
    output_data = output_data.reshape(output_data.shape[0],
    output_data.shape[1], 1)

    return [output_data, output_labels]


# Select random snippets of consistent sequence lengths from the train dataset
def select_random_sequences(input_data, input_labels, sequence_length, num_sequences):

    output_data = np.zeros([num_sequences, sequence_length])
    output_labels = np.zeros(num_sequences)

    for index in range(num_sequences):
        input_start = np.random.randint(0,len(input_data) - sequence_length * 2)
        input_end = input_start + sequence_length
        output_data[index,:] = input_data[range(input_start, input_end)]
        output_labels[index] = input_labels[input_end]

    return [output_data, output_labels]


def define_model(input_x, input_y):

	model = Sequential()
	model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(np.shape(input_x)[1],np.shape(input_x)[2]), strides=2))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(1, activation='linear'))
	model.compile(loss=keras.losses.logcosh,
				  optimizer=keras.optimizers.adam(),
				  metrics=['mae'])
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

	model.save('old_faithful_time.h5')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.show()

	return([model, history.history, history.params])


def visualize_network(model, data, labels, sequence_len, scale_factor_x, scale_factor_y, title):

	sequential_test = np.zeros([len(data)-sequence_len, sequence_len, 1])

	for i in range(len(data) - sequence_len):
		sequential_test[i, :, 0] = np.squeeze(data[i:i+sequence_len,0])

	test_preds = np.squeeze(model.predict(sequential_test)) * scale_factor_y
	test_labels = np.squeeze(labels[0:-1 - sequence_len + 1] * scale_factor_y)

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
	plt.title("Prediction Mean/Variance vs. Time to Eruption: ")
	plt.show()


data = x_test
labels = y_test
sequence_len = 150
scale_factor_x = temp_scale
scale_factor_y = time_scale



sequential_test = np.zeros([len(data)-sequence_len, sequence_len, 1])

for i in range(len(data) - sequence_len):
	sequential_test[i, :, 0] = np.squeeze(data[i:i+sequence_len,0])

test_preds = np.squeeze(model.predict(sequential_test)) * scale_factor_y
test_labels = np.squeeze(labels[0:-1 - sequence_len + 1] * scale_factor_y)

delta = np.squeeze(np.array([test_preds - test_labels]))

preds_mean = []
preds_std = []

plt.hist(test_labels,range(1, int(np.max(test_labels))))


for time in range(1, int(np.max(test_labels))):
	preds_mean.append(np.mean(delta[test_labels == time]))
	preds_std.append(np.std(delta[test_labels == time]))


plt.errorbar(range(len(preds_mean)), preds_mean, np.sqrt(preds_std), elinewidth = 0.5, capsize=0, c='r', ecolor = '#1f77b4')
plt.plot(range(int(np.min(test_labels)), int(np.max(test_labels))), np.zeros(int(np.max(test_labels))), c='gray', linewidth=0.6)
plt.xlabel("Time to Eruption (min)")
plt.ylabel("Prediction Error (min)")
#plt.xlim([95, 0])
#plt.ylim([-15, 15])
plt.title("Prediction Mean/Variance vs. Time to Eruption: ")
plt.show()



visualize_network(model, x_test, y_test, 150, temp_scale, time_scale, "Test")











def animated_plot(data, max_x, max_y, frames, interval):

	fig = plt.figure()
	ax = plt.axes(xlim = (0,max_x), ylim = (0, max_y))
	line, = ax.plot([], [])

	def init():
		line.set_data([], [])
		return line,

	def animate(i):
		line.set_data(range(max_x), data[i:i+max_x])
		return line,

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
	HTML(anim.to_html5_video())

	#anim._repr_html_() is None
	#rc('animation', html='html5')
	return anim


animated_plot(temp, 1000, 65, 1000, 1)











max_x = 1000
max_y = 65
frames = 100
interval = 10

fig = plt.figure()
ax = plt.axes(xlim = (0,max_x), ylim = (0, max_y))
line, = ax.plot([], [])

def init():
	line.set_data([], [])
	return line,

def animate(i):
	line.set_data(range(max_x), data[i:i+max_x])
	return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
HTML(anim.to_html5_video())

#anim._repr_html_() is None
#rc('animation', html='html5')
return anim


















































plt.hist(temp,100)
plt.show()

fname = 'Old_Faithful_Logger.csv'
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

[] = [time_to_eruption[0:int(test_percentage * len(time_to_eruption))], temp[0:int(test_percentage * len(temp))]]

[] = [time_to_eruption[int(test_percentage * len(time_to_eruption)) : len(time_to_eruption)], temp[int(test_percentage * len(temp)) : len(temp)]]


# First portion is input, second portion is labels


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




evaluation_len = 1000
input_length = 150
scale_factor = time_scale
time_scale

random_index = np.random.randint(0, len(time_test) - evaluation_len)
model_validation = np.zeros([evaluation_len,input_length,1])
model_validation.shape
preds = []

for i in range(evaluation_len):
	model_validation[i,:,0] = temp_test[i+random_index:input_length + random_index + i]

preds = model.predict(model_validation) * scale_factor
labels = time_test[random_index+ input_length:random_index + evaluation_len + input_length] * scale_factor
plt.plot(preds[:,0] - labels, linewidth = 0.8)
plt.plot(range(evaluation_len), np.zeros(evaluation_len), linewidth = 1, c = 'r')
#plt.plot(labels, c = 'r', linewidth = 0.8)
plt.ylim(-20, 20)
plt.xlim(0, evaluation_len)
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

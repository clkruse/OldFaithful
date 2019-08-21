import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc, cm
from IPython.display import HTML
import os
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D
from keras import backend as K
import joypy

#data from https://geysertimes.org/datalogger/archive.php
[time, temp] = import_data('Old_Faithful_Logger.csv')
np.shape(time)
#animated_plot(temp, 1000, 65, 500, 30)

time_to_eruption = create_erupt_countdown(temp, 4.0)

[temp_norm, temp_scale] = normalize_data(temp)

[time_norm, time_scale] = normalize_data(time_to_eruption)

[temp_train, time_train, temp_test, time_test] = split_train_test(temp_norm, time_norm, 0.1)

variable = []
train_losses = []
test_losses = []
for var in range(200, 201, 75):
	print("seq_len:", var)
	[x_train, y_train] = select_sequential_sequences(temp_train, time_train, var)
	[x_test, y_test] = select_sequential_sequences(temp_test, time_test, var)

	[x_train, y_train] = filter_outliers(x_train, y_train, 95)
	[x_test, y_test] = filter_outliers(x_test, y_test, 95)

	model = define_model(x_train, y_train, 1)

	[model, history, params] = train_model(model, 5, x_train, y_train, x_test, y_test)
	variable.append(var)
	train_losses.append(history['loss'][-1])
	test_losses.append(history['val_loss'][-1])
	prediction_overlay(x_test, y_test, model, time_scale, 500, var)
	visualize_network(model, x_test, y_test, var, temp_scale, time_scale, "Sequence Length " + str(var))
	plt.plot(variable, train_losses)
	plt.plot(variable, test_losses, c='r')
	plt.show()

plt.plot(variable, train_losses)
plt.plot(variable, test_losses, c='r')
plt.title("Input Sequence Length vs. Loss")
plt.xlabel("Sequence Length")
plt.ylabel("Model Loss Value")
plt.legend(["Train Loss", "Test Loss"])
plt.show()


np.max(labels)

preds, labels = prediction_overlay(x_test, y_test, model, time_scale, 26800, 200)

interval = 10
for time_val in range(0,int(np.max(labels)-interval), interval):
    indexes = np.where((labels >= time_val) & (labels  < time_val + interval))
    error = preds[indexes] - labels[indexes]
    plt.hist(error,100, density=True)
    plt.xlim([-80, 80])
    plt.grid()
    plt.axvline(np.mean(error), c='r', linewidth=1)
    plt.title("Error for times to eruption between " + str(time_val) + " and " + str(time_val + interval))
    plt.show()


x = np.arange(0,100,0.1)
y =[n*x for n in range(1,4)]
np.shape(y)


create_joyplot(preds, labels, 2)




plt.plot(preds[1000:3000]-labels[1000:3000]);plt.grid();plt.ylim([-20, 20]);plt.show()

preds_train, labels_train = prediction_overlay(x_train, y_train, model, time_scale, 241117, 200)

plt.hist(preds-labels,151);plt.xlim([-25, 25]);plt.grid();plt.title("Error Distribution in the Test Dataset");plt.show()
plt.hist(preds_train-labels_train,151);plt.xlim([-25, 25]);plt.grid();plt.title("Error Distribution in the Train Dataset");plt.show()

plt.scatter(labels_train,preds_train-labels_train, s=0.1, alpha=0.1);plt.ylabel("error"),plt.xlabel("Correct Value");plt.show()

visualize_network(model, x_test, y_test, 1500, temp_scale, time_scale, "Test")

visualize_network(model, x_train, y_train, 50, temp_scale, time_scale, "Test")



for i in range(10):
	plot_length = 1000
	random_index = np.random.randint(0, len(temp_test) - plot_length)
	plt.plot(temp_test[random_index:random_index + plot_length]-0.2, linewidth = 1)
	plt.plot(time_test[random_index:random_index + plot_length], linewidth = 1, c='r')
	plt.legend(['Input (x): Temperature', 'Output (y): Time to next eruption'])
	plt.title("Start Index: " + str(random_index))
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
	threshold = np.array(y_data * time_scale < time_threshold)
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

def define_model(input_x, input_y, var):

	model = Sequential()
	model.add(Conv1D(filters=8, kernel_size=8, activation='relu', input_shape=(np.shape(input_x)[1],np.shape(input_x)[2]), strides=1))
	model.add(MaxPooling1D(pool_size=2))
	#model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(1, activation='linear'))
	model.compile(loss=keras.losses.mean_absolute_error,
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
	                  batch_size=256,
	                  shuffle=True)

	model.save('old_faithful_time.h5')
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


def prediction_overlay(x_test, y_test, model, time_scale, evaluation_len, sequence_len):

	random_index = np.random.randint(0, len(x_test) - evaluation_len)
	model_validation = np.zeros([evaluation_len,sequence_len,1])
	preds = []

	for i in range(evaluation_len):
		model_validation[i,:,0] = x_test[i+random_index,:,0]

	preds = model.predict(model_validation) * time_scale
	labels = y_test[random_index:random_index + evaluation_len] * time_scale

	plt.plot(preds[:,0], linewidth = 0.8)
	plt.plot(labels, linewidth = 0.8, c='r')
	# Plot line at 0
	#plt.plot(range(evaluation_len), np.zeros(evaluation_len), linewidth = 1, c = 'r')
	#plt.ylim(-10, 90)
	plt.grid()
	plt.xlim(0, evaluation_len)
	plt.legend(["Predictions", "Labels"])
	plt.title("Start Index: " + str(random_index))
	plt.show()
	return preds[:,0], labels

def create_joyplot(preds, labels, interval):
    interval = interval
    errors = []
    timespan = []
    error_colors = []
    for time_val in range(0,int(np.max(labels)-interval), interval):
        indexes = np.where((labels >= time_val) & (labels  < time_val + interval))
        error = preds[indexes] - labels[indexes]
        errors.append(error)
        error_colors.append(np.median(error))
        if time_val % 12 == 0:
            timespan.append(str(time_val) + " min")
        else: timespan.append(None)


    orig_cmap = cm.RdYlBu
    # Create a normalization scheme so that colorbar has correct legend
    norm = plt.Normalize(np.min(error_colors), np.max(error_colors))
    # Create a colormap based on avg error at each interval
    error_cmap = ListedColormap(orig_cmap(norm(error_colors)))
    # Create a color bar
    sm = cm.ScalarMappable(cmap=orig_cmap, norm=norm)
    sm.set_array([])

    fig, axes = joypy.joyplot(errors, labels=timespan, x_range=[-30, 15], overlap=1, figsize=(6,5), colormap=error_cmap, title="Distribution of Error Grouped by Time to Eruption", fade=False, linewidth=0.5,range_style='all', grid='x')
    fig.colorbar(sm, ax=axes, label="Median Error (minutes)")
    plt.xlabel("Error (min)")
    ax = axes[-1]
    ax.yaxis.set_label_position("left")
    ax.set_ylabel("Time until next eruption (min)")
    ax.yaxis.set_label_coords(-0.18,0.5)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_visible(True)
    plt.show()
    fig.savefig("Distribution of Error by Time to Eruption.png", dpi=300, bbox_inches='tight', pad_inches=0.1)


train_losses = np.array([0.0030, 0.0027, 0.0026, 0.0026, 0.0027, 0.0026, 0.0025, 0.0026, 0.0025, 0.0025, 0.0028, 0.0026, 0.0025, 0.0024, 0.0026, 0.0026, 0.0026, 0.0025, 0.0026])

test_losses = np.array([0.0029, 0.0029, 0.0030, 0.0027, 0.0027, 0.0024, 0.0027, 0.0027, 0.0028, 0.0027, 0.0028, 0.0024, 0.0027, 0.0027, 0.0030, 0.0026, 0.0033, 0.0028, 0.0030])



variable = []
for i in range(25, 1000, 75):
	variable.append(i)

variable = np.array(variable)






plt.plot(variable, train_losses)
plt.plot(variable, test_losses, c='r')
plt.legend(["Train Losses", "Test Losses"])
plt.grid()
plt.title("Losses vs. Input Sequence Length")
plt.xlabel("Sequence Length")
plt.figtext(0.5, -0.1, "Arch: 8c8-stride1-relu+mp2+2(32n-relu)+drop0.1+32n-relu+1n-linear.\n3 epochs with a batch size of 256", horizontalalignment='center', verticalalignment='bottom', fontstyle='italic')
plt.ylabel("Loss Value (MSE)")
plt.savefig("Loss vs. Sequence Length.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


plt.plot(time_to_eruption[1350:1850])
plt.title("Time until Next Eruption")
plt.grid(linewidth=0.2)
plt.xlabel("Elapsed Time (min)")
plt.ylabel("Time until Eruption (min)")
plt.rc('axes', linewidth=0.5)
plt.figtext(0.5, -0.05, "Data from geysertimes.org", horizontalalignment='center', verticalalignment='bottom', fontstyle='italic')
plt.savefig("Time until Next Eruption.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()





interval = np.genfromtxt("old_faithful_interval.csv", delimiter = ',')
np.size(interval)
train = interval[0 : -1 - int(0.1 * np.size(interval))]
test = interval[np.size(interval) - int(0.1 * np.size(interval)) : -1]

seq_len = 15
[x_train, y_train] = select_sequential_sequences(train, train, seq_len)
[x_test, y_test] = select_sequential_sequences(test, test, seq_len)

model = define_model(x_train, y_train, 1)

[model, history, params] = train_model(model, 100, x_train, y_train, x_test, y_test)

np.shape(y_test)
preds, labels = prediction_overlay(x_train, y_train, model, 1, 50000, 200)

plt.plot(preds,linewidth=0.5)
plt.plot(labels-10, c='r',linewidth=0.5)
plt.show()

diff = []
for i in range(800,1100):
    diff = np.append(diff, np.median(i/100+preds-labels))

plt.plot(diff)
plt.grid()
plt.hist(9.2+preds-labels, 100)
plt.show()

plt.scatter(9.2+preds, labels, s=0.2, alpha=0.1)
plt.plot(range(40,250),range(40,250), c='r')
plt.xlim([80, 100])
plt.ylim([80, 100])
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.show()

a = plt.hist(interval,250, alpha = 0.5)
counts = a[0]
bins = a[1]
bins = bins[0:-1]
plt.plot(bins+0.5, counts, c='r')
plt.xlabel("Time Between Eruptions (min)")
plt.ylabel("Number of Eruptions")
plt.title("Distribution of Old Faithful Eruption Times")
plt.xlim([45, 135])
plt.figtext(0.5, -0.1, "Note the bimodal peaks centered at 65 min and 92 min\nEruptions between 2000 and 2011. Data from geyserstudy.org", horizontalalignment='center', verticalalignment='bottom', fontstyle='italic')
plt.rc('axes', linewidth=0.5)
plt.axvline(np.mean(interval[interval > 75]), c='gray', linewidth=0.5)
plt.axvline(np.mean(interval[interval < 75]), c='gray', linewidth=0.5)
plt.savefig("Distribution of Eruption Times.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


np.mean(interval[interval < 75])
time[-1]

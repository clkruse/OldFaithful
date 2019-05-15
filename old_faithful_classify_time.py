import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import keras
from matplotlib import animation, rc
from IPython.display import HTML
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras import backend as K


data = DataPreparation()
data.import_data('Old_Faithful_Logger.csv')
data.create_erupt_countdown(data.temp, 2.0)
data.norm_temp = data.normalize_data(data.temp)
data.norm_time = data.normalize_data(data.countdown)
data.select_sequential_sequences(data.norm_temp, data.norm_time, sequence_len=150)
#data.filter_outliers(data.x, data.y, 100)
for item in data.__dict__.keys(): print(item)

network = Training(data.x_filtered, data.y_filtered)
network.define_model(network.train_x, network.train_y)
network.train_model(network.model, 2)







# This class contains all data and preprocessing steps for model training
class DataPreparation(object):

    # I don't believe I need to initialize anything
    def __init__(self):
        pass

    # Import from a csv file
    def import_data(self, fname):
        # Import old faithful temperature/time data
        # Dataset stretches from 4/13/2015 - 10/26/2015
        # Temperature in degrees C
        # Time measured once per minute. Time in Unix seconds
        data = np.genfromtxt(fname, delimiter = ',')
        self.time = data[1:-1,0]
        self.temp = data[1:-1,1]

        return self.temp, self.time

    # Normalize data simplistic. Will want to scale from arbitrary ranges at
    # some point. For now, this normalization simply scales the max value to 1.0
    def normalize_data(self, data):
        min = np.min(data)
        max = np.max(data)
        self.norm_data = data / max
        self.scale_factor = max
        return self.norm_data, self.scale_factor


    # Create the time to eruption vector. For each temperature, there is an associated time until the next eruption
    def create_erupt_countdown(self, data, threshold_std):
        # Create a vector of delta temperatures
        delta = data[1:len(data)] - data[0:len(data)-1]
        delta = np.insert(delta, 0, 0)
        threshold = threshold_std * np.std(delta) + np.mean(delta)
        eruptions = delta > threshold
        print(sum(eruptions), "eruptions are detected at a threshold of", threshold_std)

        countdown = []
        counter = 0
        temp_std = np.std(data)
        temp_mean = np.mean(data)
        for value in range(len(eruptions)-1,-1,-1):
            if data[value] >=  temp_std * 0.8 + temp_mean:
                countdown.append(0)

            else:
                countdown.append(counter)
            counter += 1

            if eruptions[value] == True:
                counter = 0

        self.countdown = np.array(countdown[::-1])
        return self.countdown


    # Split the train and test datasets
    # Keras does this already, so it may be unnecessary
    def split_train_test(self, data, labels, percentage):
        train_percentage = 1 - percentage

        [self.train_x, self.train_y] = [data[0:int(train_percentage * len(data))], labels[0:int(train_percentage * len(labels))]]

        [self.test_x, self.test_y] = [data[int(train_percentage * len(data)) : len(data)], labels[int(train_percentage * len(labels)) : len(labels)]]

        print("Number of training samples:", len(self.train_x))
        print("Number of test samples:", len(self.test_x))

        return self.train_x, self.train_y, self.test_x, self.test_x


    # Select sequences with a sliding window across the dataset
    def select_sequential_sequences(self, input_data, input_labels, sequence_len):

        output_data = np.zeros([len(input_data)-sequence_len, sequence_len])
        self.y = input_labels[sequence_len - 1:-1]

        for i in range(len(input_data) - sequence_len):
            output_data[i, :] = input_data[i:i+sequence_len]

        # Reshape data to make keras happy??
        self.x = output_data.reshape(output_data.shape[0],
        output_data.shape[1], 1)

        return self.x, self.y


    # Select random snippets of consistent sequence lengths from the train dataset
    def select_random_sequences(self, input_data, input_labels, sequence_len, num_sequences):

        self.x = np.zeros([num_sequences, sequence_len])
        self.y = np.zeros(num_sequences)

        for index in range(num_sequences):
            input_start = np.random.randint(0,len(input_data) - sequence_len * 2)
            input_end = input_start + sequence_len
            self.x[index,:] = input_data[range(input_start, input_end)]
            self.y[index] = input_labels[input_end]

        return self.x, self.y

    #Filter outlier inputs based on the time exceeding a threshold value
    def filter_outliers(self, x_data, y_data, time_threshold):
        threshold = np.array(y_data < time_threshold)
        print("Number of samples removed", len(y_data) - sum(threshold))
        self.x_filtered = x_data[threshold, :, :]
        self.y_filtered = y_data[threshold]

        return self.x_filtered, self.y_filtered


# This class contains functions for defining and training a models
# Key elements that it returns are the model itself,
# training history, and parameters
class Training(object):

    # I don't believe I need to initialize anything
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def define_model(self, train_x, train_y):

        self.model = Sequential()
        self.model.add(Conv1D(filters=4,
                        kernel_size=16,
                        activation='relu',
                        strides=4,
                        padding='valid', input_shape=(np.shape(train_x)[1],np.shape(train_x)[2])))
        #model.add(MaxPooling1D(pool_size=2))
        #model.add(Conv1D(filters=16, kernel_size=4, activation='relu', strides=4))
        #model.add(MaxPooling1D(pool_size=2))
        #model.add(Dense(16, activation='relu',input_shape=(np.shape(input_x)[1],np.shape(input_x)[2])))
        self.model.add(Conv1D(filters=4,
                        kernel_size=16,
                        activation='relu',
                        strides=4,
                        padding='valid'))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='softmax'))
        self.model.compile(loss=keras.losses.msle,
                      optimizer=keras.optimizers.adam(),
                      metrics=[keras.losses.mean_squared_error])
        return self.model

    def train_model(self, model, epochs):
        #model = load_model('old_faithful_time.h5')
        history = model.fit(self.train_x,
                          self.train_y,
                          epochs=epochs,
                          validation_split = 0.1,
                          verbose=1,
                          batch_size=32,
                          shuffle=True)

        #model.save('old_faithful_time.h5')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], c='r')
        plt.legend(["Train Loss", "Test Loss"])
        plt.show()


        return self.model, history.history, history.params


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
    pred_std = np.zeros(len(preds))
    label_max = np.zeros(len(preds))

    for i in range(len(preds)):
        #pred_max[i] = np.argmax(preds[i])
        pred_max[i] = np.sum(preds[i] * range(174))
        pred_std[i] = np.std(preds[i][preds[i] > 0]) * preds.shape[1]
        label_max[i] = np.argmax(y_test[i])
        diff = pred_max[i] - label_max[i]
        deltas[i] = diff

    random_index = np.random.randint(0, len(x_test) - evaluation_len)

    plt.plot(label_max[random_index:random_index + evaluation_len])
    #plt.plot(deltas[random_index:random_index + evaluation_len], c='grey')
    plt.errorbar(range(evaluation_len), pred_max[random_index:random_index + evaluation_len], pred_std[random_index:random_index + evaluation_len], elinewidth = 0.5, capsize=0, c = 'r', ecolor = 'grey')
    #plt.plot(pred_max[random_index:random_index + evaluation_len], c='r')
    #plt.legend(["Delta", "Labels", "Predictions"])
    plt.legend(["Labels", "Predictions"])
    plt.ylim([-20, 100])
    plt.grid()
    plt.show()



for i in range(1500, 1600):
    plt.plot(preds[i], linewidth=0.5)
    plt.plot(y_test[i]*np.max(preds[i]), c='black')
    plt.ylim([0, np.max(preds[i]) + 0.1 * np.max(preds[i])])
    plt.grid()
    plt.show()


deltas = np.zeros(len(preds))
pred_max = np.zeros(len(preds))
pred_std = np.zeros(len(preds))
label_max = np.zeros(len(preds))

for i in range(len(preds)):
    pred_max[i] = np.argmax(preds[i])
    pred_std[i] = np.std(preds[i]) * preds.shape[1]
    label_max[i] = np.argmax(y_test[i])
    diff = pred_max[i] - label_max[i]
    deltas[i] = diff

plt.errorbar(range(100), pred_max[0:100], pred_std[0:100], elinewidth = 0.5, capsize=0, ecolor = 'grey')
plt.show()

np.sum(preds[30] * range(174))
np.argmax(preds[30])
random_index = np.random.randint(0, len(predictions) - evaluation_len)

plt.plot(label_max[0:500])
plt.plot(deltas[0:500], c='grey')
plt.plot(pred_max[0:500], c='r')
plt.legend(["Labels", "Predictions", "Delta"])
plt.grid()
plt.show()

np.argmax(y_test[0:30])

plt.plot(preds[30])
preds = model.predict(x_test)
preds[1]
np.std(preds[10][preds[10] > 0.02]) * preds.shape[1]
plt.plot(preds[0])
plt.show()

plt.plot(preds[10][preds[10] > 0.02])

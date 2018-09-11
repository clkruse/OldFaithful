import numpy as np
from scipy.stats import norm
from scipy.misc import imsave
import matplotlib.pyplot as plt
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from numpy import genfromtxt

deltas = genfromtxt(fname = '/Users/ckruse/Downloads/faithful.csv', delimiter = ',')

#Normalize deltas between -1 and 1

print(np.max(deltas))

delt = deltas/np.max(deltas)*2 - 1
print(np.max(delt))

numSequences = 50000
seqLen = 25
sequences = np.zeros([numSequences, seqLen])

for i in range(0,numSequences):
    seed = np.random.randint(seqLen,np.size(delt)-seqLen)
    sequences[i,:] = delt[seed-seqLen:seed]

#Split up the time series vector into an x and y
x = sequences[:,0:seqLen-1]
y = sequences[:,seqLen-1]
x = x.reshape(x.shape[0], x.shape[1],1)
y = y.reshape(y.shape[0])

print(x.shape)
print(y.shape)

from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten

model = Sequential()
#model.add(Conv2D(8, kernel_size=(1, 3),
#                 activation='relu', padding='same',
#                 input_shape=x.shape))

#model.add(Conv2D(8, (1, 3), activation='relu', padding='same'))

#model.add(Flatten())
model.add(Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(24,1), strides=1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse'])

from keras.models import load_model

epochs = 100

#model = load_model('/Users/ckruse/Documents/python/oldFaithful.h5')
history = model.fit(x,
                    y,
                    epochs=epochs,
                    verbose=1,
                    batch_size=32,
                    validation_split=0.1,
                    shuffle=True)

model.save('/Users/ckruse/Documents/python/oldFaithful.h5')
plt.plot(history.history['loss'])
plt.show()

plt.plot(history.history['val_loss'])
plt.show()

#score = model.evaluate(data_stack, label_stack, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#model.save('/Users/ckruse/Documents/python/soccer/model.h5')

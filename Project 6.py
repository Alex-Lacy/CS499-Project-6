from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import gzip


def read_gz_file(path):
    if os.path.exists(path):
        with gzip.open(path, 'r') as pf:
            for line in pf:
                yield line
    else:
        print('the path [{}] is not exist!'.format(path))

X=[]
y=[]

con = read_gz_file('zip.train.gz')
if getattr(con, '__iter__', None):
    i=0
    for line in con:
        data = line.decode('utf-8')
        data = np.array(data.split(), dtype=float)
        y.append(data[0])
        X.append(data[1:].reshape(16,16,1))
        i+=1

X=np.array(X)
y=np.array(y)

num_classes = 10

input_shape = (16, 16, 1)
model1 = keras.Sequential()
model1.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
model1.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
model1.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model1.add(keras.layers.Dropout(0.25))
model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dropout(0.5))
model1.add(keras.layers.Dense(num_classes, activation='softmax'))

model1.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
his = model1.fit(X, y, epochs=20, validation_split=0.2, verbose=2)

history_dict = his.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
b1 = val_loss.index(min(val_loss)) + 1
epochs = range(1, len(acc) + 1)
plt.plot(b1, val_loss[b1 - 1], 'ro')
# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'b', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
print('\n', b1)
print('\n', val_loss[b1 - 1])
plt.show()

model2 = keras.Sequential()
model2.add(keras.layers.Flatten(input_shape=input_shape))
model2.add(keras.layers.Dense(270, activation='relu'))
model2.add(keras.layers.Dense(270, activation='relu'))
model2.add(keras.layers.Dense(128, activation='relu'))
model2.add(keras.layers.Dense(num_classes, activation='softmax', use_bias='FALSE'))

model1.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
his = model1.fit(X, y, epochs=20, validation_split=0.2, verbose=2)

history_dict = his.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
b1 = val_loss.index(min(val_loss)) + 1
epochs = range(1, len(acc) + 1)
plt.plot(b1, val_loss[b1 - 1], 'ro')
# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'b', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
print('\n', b1)
print('\n', val_loss[b1 - 1])
plt.show()
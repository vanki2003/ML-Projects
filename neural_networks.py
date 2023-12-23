import numpy as np
import pandas as pd
import tensorflow as tf
import keras as keras
import matplotlib.pyplot as plt
from numpy import array  
from numpy import argmax  
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

data = pd.read_excel('datasets\Dry_Bean_Dataset.xlsx',engine='openpyxl')
data.head()
data.shape

x_data = data.drop(columns='Class')
y_data = data['Class']

data = data.to_numpy()
print(data)

y_data = np.zeros(y_data.shape, int)

encoded=tf.keras.utils.to_categorical(y_data-1, num_classes = 7)
encoded.shape

# standardizing feature values
np.mean(x_data, axis=0)
np.std(x_data)

formula = (x_data - np.mean(x_data, axis = 0)) / np.std(x_data, axis=0)
print(formula)

plt.boxplot(formula)

x_train, x_test, y_train, y_test = train_test_split(x_data, encoded, test_size=3500,random_state=1)
print( x_train, y_train, x_test, y_test)

# creating, compiling and training the neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=7, activation='softmax'))

print(model.summary())

lr = tf.keras.optimizers.Adam(learning_rate=0.01)
mt = tf.keras.metrics.CategoricalAccuracy()

model.compile(optimizer = lr, loss = tf.keras.losses.categorical_crossentropy , metrics = mt)

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# plotting loss and accuracy changes
plt.figure(figsize=[10,5])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'blue', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)

# Plotting the accuracy curve
plt.figure(figsize=[10,5])
plt.plot(history.history['categorical_accuracy'], 'black', linewidth=2.0)  
plt.plot(history.history['val_categorical_accuracy'], 'blue', linewidth=2.0)  
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
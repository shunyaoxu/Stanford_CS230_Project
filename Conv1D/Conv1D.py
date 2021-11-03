import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io

#%% Part1 Data Processing
dataset1 = scipy.io.loadmat('E:\Stanford_Autumn_2021\CS_230\Project\Dataset\Data_HM\Data_HM\X_Signal\HMNYXZ_signal.mat')
dataset2 = scipy.io.loadmat('E:\Stanford_Autumn_2021\CS_230\Project\Dataset\Data_HM\Data_HM\X_Signal\HMNYZX_signal.mat')
labelset1 = scipy.io.loadmat('E:\Stanford_Autumn_2021\CS_230\Project\Dataset\Data_HM\Data_HM\Y_MPS\HMNYXZ_Y.mat')
labelset2 = scipy.io.loadmat('E:\Stanford_Autumn_2021\CS_230\Project\Dataset\Data_HM\Data_HM\Y_MPS\HMNYZX_Y.mat')

signal_matrix = np.concatenate((dataset1['signal_matrix'], dataset2['signal_matrix']), axis=2)
#print("shape", signal_matrix.shape)
zeros = np.zeros((65,8,signal_matrix.shape[2]))
#print("shape", zeros.shape)
input_matrix = np.concatenate((zeros, signal_matrix), axis=0)
input_matrix = np.concatenate((input_matrix, zeros), axis=0).T
X = input_matrix.reshape(input_matrix.shape[0],-1)
label = np.concatenate((labelset1['label'], labelset2['label']), axis=0)
print("Input Shape", X.shape)
print("Label Shape", label.shape)
#plt.plot(np.arange(1,201), input_matrix[:,0,0])

#%% Part2 Split into train and dev set
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

X = normalize(X, axis=0, norm='max')
X_train, X_dev, y_train, y_dev = train_test_split(X, label, test_size=0.2)
print(f"X_train = {X_train.shape}")
print(X_train)
print(f"y_train = {y_train.shape}")
print(y_train)
print(f"X_dev = {X_dev.shape}")
print(X_dev)
print(f"y_dev = {y_dev.shape}")
print(y_dev)


#%% Part3 Model Construction and Training
# Conv1D
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MaxPooling1D, Conv1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPool2D, Activation

# The number of steps within one time segment
num_classes = y_train.shape[1]
kernel_size=2

input1 = keras.Input(shape=(1600,1))
x = Conv1D(64, kernel_size, activation='relu')(input1)
x = Conv1D(64, kernel_size, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(1000, activation='relu')(x)
x = Dense(100, activation='relu')(x)
output1 = Dense(num_classes)(x)
model_conv1D = tf.keras.Model(inputs=input1, outputs=output1)

model_conv1D.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])

print(model_conv1D.summary())

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_dev_reshaped = np.reshape(X_dev, (X_dev.shape[0], X_dev.shape[1], 1))

history = model_conv1D.fit(X_train_reshaped, y_train, epochs=200,
          batch_size=100, verbose=1,
          validation_data=(X_dev_reshaped, y_dev))

#tf.saved_model.save(model_conv1D, "E:/RPI_Spring 2020/Research/GridX/New Folder/1DModelOpwithTRT")

def printHistory(history):
       loss_curve = history.history["loss"]
       acc_curve = history.history["accuracy"]

       loss_val_curve = history.history["val_loss"]
       acc_val_curve = history.history["val_accuracy"]

       fig, axs = plt.subplots(figsize=(14,6))
       axs.plot(acc_curve, color='royalblue', alpha = 1, linewidth=3.0, label="Train")
       axs.plot(acc_val_curve, color='red', alpha = 1, linewidth=3.0, label="Validation")
       axs.set_ylabel('Accuracy', fontsize=20)
       axs.set_xlabel('Epoch', fontsize=20)
       axs.grid(color='g', ls = '-.', lw = 0.3)
       plt.xticks(fontsize = 16)
       plt.yticks(fontsize = 16)
       plt.legend(loc='lower right', fontsize=20)
       plt.show()
printHistory(history)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io

#%% Part1 Data Processing
dataset1 = scipy.io.loadmat('.\Data_HM\X_Signal\HMNYXZ_signal.mat')
dataset2 = scipy.io.loadmat('.\Data_HM\X_Signal\HMNYZX_signal.mat')
dataset3 = scipy.io.loadmat('.\Data_HM\X_Signal\HMXNYZ_signal.mat')
dataset4 = scipy.io.loadmat('.\Data_HM\X_Signal\HMXYZ_signal.mat')
dataset5 = scipy.io.loadmat('.\Data_HM\X_Signal\HMXZNY_signal.mat')
dataset6 = scipy.io.loadmat('.\Data_HM\X_Signal\HMXZY_signal.mat')
dataset7 = scipy.io.loadmat('.\Data_HM\X_Signal\HMYXZ_signal.mat')
dataset8 = scipy.io.loadmat('.\Data_HM\X_Signal\HMYZX_signal.mat')
dataset9 = scipy.io.loadmat('.\Data_HM\X_Signal\HMZNYX_signal.mat')
dataset10 = scipy.io.loadmat('.\Data_HM\X_Signal\HMZXNY_signal.mat')
dataset11 = scipy.io.loadmat('.\Data_HM\X_Signal\HMZXY_signal.mat')
dataset12 = scipy.io.loadmat('.\Data_HM\X_Signal\HMZYX_signal.mat')
labelset1 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMNYXZ_Y.mat')
labelset2 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMNYZX_Y.mat')
labelset3 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMXNYZ_Y.mat')
labelset4 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMXYZ_Y.mat')
labelset5 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMXZNY_Y.mat')
labelset6 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMXZY_Y.mat')
labelset7 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMYXZ_Y.mat')
labelset8 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMYZX_Y.mat')
labelset9 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMZNYX_Y.mat')
labelset10 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMZXNY_Y.mat')
labelset11 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMZXY_Y.mat')
labelset12 = scipy.io.loadmat('.\Data_HM\Y_MPS\HMZYX_Y.mat')

signal_matrix = np.concatenate((dataset1['signal_matrix'], dataset2['signal_matrix'],
                                dataset3['signal_matrix'], dataset4['signal_matrix'],
                                dataset5['signal_matrix'], dataset6['signal_matrix'],
                                dataset7['signal_matrix'], dataset8['signal_matrix'],
                                dataset8['signal_matrix'], dataset10['signal_matrix'],
                                dataset11['signal_matrix'], dataset12['signal_matrix']), axis=2)
#print("shape", signal_matrix.shape)
zeros = np.zeros((65,8,signal_matrix.shape[2]))
#print("shape", zeros.shape)
input_matrix = np.concatenate((zeros, signal_matrix), axis=0)
input_matrix = np.concatenate((input_matrix, zeros), axis=0).T
X = input_matrix.reshape(input_matrix.shape[0],-1)
label = np.concatenate((labelset1['label'], labelset2['label'],
                        labelset3['label'], labelset4['label'],
                        labelset5['label'], labelset6['label'],
                        labelset7['label'], labelset8['label'],
                        labelset9['label'], labelset10['label'],
                        labelset11['label'], labelset12['label']), axis=0)
print("Input Shape", X.shape)
print("Label Shape", label.shape)

#%% Part2 Split into train and dev set
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

X = normalize(X, axis=0, norm='max')
X_train, X_dev, y_train, y_dev = train_test_split(X, label, test_size=0.2)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5)
print(f"X_train = {X_train.shape}")
print(X_train)
print(f"y_train = {y_train.shape}")
print(y_train)
print(f"X_dev = {X_dev.shape}")
print(X_dev)
print(f"y_dev = {y_dev.shape}")
print(y_dev)
print(f"X_test = {X_test.shape}")
print(X_dev)
print(f"y_test = {y_test.shape}")
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
x = Dense(400, activation='relu')(x)
x = Dense(100, activation='relu')(x)
output1 = Dense(num_classes)(x)
model_conv1D = tf.keras.Model(inputs=input1, outputs=output1)
model_conv1D.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['MeanSquaredError'])

print(model_conv1D.summary())

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_dev_reshaped = np.reshape(X_dev, (X_dev.shape[0], X_dev.shape[1], 1))

history = model_conv1D.fit(X_train_reshaped, y_train, epochs=100,
          batch_size=128, verbose=1,
          validation_data=(X_dev_reshaped, y_dev))


def printHistory(history):
       loss_curve = history.history["loss"]
       MSE_curve = history.history["mean_squared_error"]

       loss_val_curve = history.history["val_loss"]
       MSE_val_curve = history.history["val_mean_squared_error"]

       fig, axs = plt.subplots(figsize=(14,6))
       axs.plot(MSE_curve, color='royalblue', alpha = 1, linewidth=3.0, label="Train")
       axs.plot(MSE_val_curve, color='red', alpha = 1, linewidth=3.0, label="Validation")
       axs.set_ylabel('Mean Squared Error', fontsize=20)
       axs.set_xlabel('Epoch', fontsize=20)
       axs.grid(color='g', ls = '-.', lw = 0.3)
       plt.xticks(fontsize = 16)
       plt.yticks(fontsize = 16)
       plt.legend(loc='lower right', fontsize=20)
       plt.show()

       fig2, axs2 = plt.subplots(figsize=(14, 6))
       axs2.plot(loss_curve, color='royalblue', alpha=1, linewidth=3.0, label="loss")
       axs2.plot(loss_val_curve, color='red', alpha=1, linewidth=3.0, label="val_loss")
       axs2.set_ylabel('Loss', fontsize=20)
       axs2.set_xlabel('Epoch', fontsize=20)
       axs2.grid(color='g', ls='-.', lw=0.3)
       plt.xticks(fontsize=16)
       plt.yticks(fontsize=16)
       plt.legend(loc='lower right', fontsize=20)
       plt.show()

printHistory(history)
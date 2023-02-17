#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LayerNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam 
import tensorflow_model_optimization as tfmot
import os

from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras import regularizers as reg
RANDOM_SEED = 42


# # A Look At the DataSet

# Each of the files represents a test subject that performs the 18 activities for 3 min intervals. Depending on the directory of the file, the X, Y, Z columns represent the 3-axis readings to the corresponding sensor (Accelerometer or Gyroscope).

# In[3]:


accel = pd.read_csv('accel/data_1600_accel_watch.txt', names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
accel


# In[4]:


accelWalking = accel.where(accel['Activity Label'] == 'A').dropna()

#Remove Semicolon from ends of z-column
accelWalking['Z'] = accelWalking['Z'].str[0: -1].astype(float)

print("ACCELEROMETER SPECTOGRAMS FOR USER 0 WALKING")

plt.plot(np.arange(len(accelWalking)), accelWalking['X'], color='red', marker='o')
plt.title('AccelX Vs Timestep', fontsize=14)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('accelX', fontsize=14)
plt.grid(True)
plt.show()

plt.plot(np.arange(len(accelWalking)), accelWalking['Y'], color='red', marker='o')
plt.title('AccelY Vs Timestep', fontsize=14)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('accelY', fontsize=14)
plt.grid(True)
plt.show()

plt.plot(np.arange(len(accelWalking)), accelWalking['Z'], color='red', marker='o')
plt.title('AccelZ Vs Timestep', fontsize=14)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('accelZ', fontsize=14)
plt.grid(True)
plt.show()


# This file contains the gyroscope readings of the user 1600

# In[5]:


gyro = pd.read_csv('gyro/data_1603_gyro_watch.txt', names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
gyro


# In[6]:


gyroWalking = gyro.where(accel['Activity Label'] == 'A').dropna()

#Remove Semicolon from ends of z-column
gyroWalking['Z'] = gyroWalking['Z'].str[0: -1].astype(float)

print("GYROSCOPE SPECTOGRAMS FOR USER 0 WALKING")

plt.plot(np.arange(len(gyroWalking)), gyroWalking['X'], color='red', marker='o')
plt.title('GyroX Vs Timestep', fontsize=14)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('gyroX', fontsize=14)
plt.grid(True)
plt.show()

plt.plot(np.arange(len(gyroWalking)), gyroWalking['Y'], color='red', marker='o')
plt.title('GyroY Vs Timestep', fontsize=14)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('gyroY', fontsize=14)
plt.grid(True)
plt.show()

plt.plot(np.arange(len(gyroWalking)), gyroWalking['Z'], color='red', marker='o')
plt.title('GyroZ Vs Timestep', fontsize=14)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('gyroZ', fontsize=14)
plt.grid(True)
plt.show()


# Lets take a look at the distribution of the dataset

# In[11]:


# Taking a Look at the ditribution of training data labels
accelDirs = list(os.scandir('accel'))[2:53]
del accelDirs[37:41]
gyroDirs = list(os.scandir('gyro'))[2:53]
del gyroDirs[37:41]
labels = ['A', 'B', 'C', 'D', 'E']
overallDist = [0,0,0,0,0]
allDirs = list(zip(accelDirs, gyroDirs))
trainingIndices = np.random.choice(len(allDirs), 35, replace=False).tolist()
trainingDirs = [allDirs[i] for i in range(len(allDirs)) if i in trainingIndices]
for directory in trainingDirs:
    accelDf = pd.read_csv(directory[0], names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    gyroDf = pd.read_csv(directory[1], names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    for i in range(len(labels)):
        overallDist[i]  += len(accelDf.where(accelDf['Activity Label'] == labels[i]).dropna())
        overallDist[i] += len(gyroDf.where(gyroDf['Activity Label'] == labels[i]).dropna())
    
plt.pie(np.asarray(overallDist), labels=["Walking", "Jogging", "Stairs", "Standing", "Sitting"])
plt.legend(title="Activities: ")
plt.show()


# # Splitting Dataset

# We randomly assign 70 % of the users to the training set and the remaining users to the test set. We excluded users 1637 to 1641 because they data suggests that these subjects did not follow the experiment guidelines.

# In[12]:


accelDirs = list(os.scandir('accel'))[2:53]
del accelDirs[37:41]
gyroDirs = list(os.scandir('gyro'))[2:53]
del gyroDirs[37:41]
#corresponding accel and gyro files per user
allDirs = list(zip(accelDirs, gyroDirs))
#randomly assign users to training set
trainingIndices = np.random.choice(len(allDirs), 35, replace=False).tolist()
trainingDirs = [allDirs[i] for i in range(len(allDirs)) if i in trainingIndices]
print("Users in Training Set:\n")
trainingIndices.sort()
for index in trainingIndices:
    print("User " + str(index))


# In[13]:


#rest go to test set
testDirs = [x for x in allDirs if x not in trainingDirs]
print("Users in Test Set:\n")
for index in [x for x in range(len(allDirs)) if x not in trainingIndices]:
    print("User " + str(index))


# Normalization and Augmentation Functions

# In[14]:


#Normalize relative to the column mean
def normalize_window(df):
    df['X_x'] = (df['X_x'] - df['X_x'].mean())/df['X_x'].astype(float).std()
    df['Y_x'] = (df['Y_x'] - df['Y_x'].mean())/df['Y_x'].astype(float).std()
    df['Z_x'] = (df['Z_x'] - df['Z_x'].mean())/df['Z_x'].astype(float).std()
    df['X_y'] = (df['X_y'] - df['X_y'].mean())/df['X_y'].astype(float).std()
    df['Y_y'] = (df['Y_y'] - df['Y_y'].mean())/df['Y_y'].astype(float).std()
    df['Z_y'] = (df['Z_y'] - df['Z_y'].mean())/df['Z_y'].astype(float).std()
    
#augmentation strategy to permute the timesteps of each window
def permutation(x, max_segments=20):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


# We are only including 5 of the 18 activities: Walking, Jogging, Standing, Stairs, and Laying.

# In[15]:


lb = LabelBinarizer()
lb.fit(labels)
#EXAMPLE OF OUTPUT
lb.transform(['A'])


# # Setting Up Training Data

# Create a series of windows for each activity that a user performs. For now, the code regarding the merging of the accelerometer data and gyroscope data has been commented out because the gyroscope data had little bearing on accuracy and loss.

# In[16]:


def userBatches(accelFile, gyroFile, windowSize, stride, windows, vectorizedActivities):
    accel = pd.read_csv(accelFile, names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    gyro = pd.read_csv(gyroFile, names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    accel['Z'] = accel['Z'].str[0: -1]
    gyro['Z'] = gyro['Z'].str[0: -1]
    for label in labels:
        #merge accel and gyroscope data to have 6 channels
        filt1 = accel.where(accel['Activity Label'] == label).dropna().sort_values('Time Stamp').iloc[:, [3,4,5]]
        filt2 = gyro.where(gyro['Activity Label'] == label).dropna().sort_values('Time Stamp').iloc[:, [0,3,4,5]]
        merged = filt1.reset_index().merge(filt2.reset_index(), left_index=True, right_index=True, how='left').dropna()
        merged = merged.drop(['index_x','index_y', 'user id'], axis=1)
        for i in range(0, merged.shape[0] - windowSize, stride):
            newWindow = merged.iloc[i:i+windowSize, :].astype(float)
            normalize_window(newWindow)
            newWindow = newWindow.to_numpy().tolist()
            ctgrs = lb.transform([label]).tolist()[0]
            vectorizedActivities.append(ctgrs)
            windows.append(newWindow)


# Following cell creates 200 timestep windows that slide 20 timesteps at each iteration

# In[17]:


N_TIME_STEPS = 200
N_FEATURES = 6
step = 20

X_train = []
y_train = []
X_test = []
y_test = []

#CREATING BATCHES OF WINDOWS FOR EACH USER IN TEST & TRAIN SET
for accelFile, gyroFile in trainingDirs:
    userBatches(accelFile, gyroFile, N_TIME_STEPS, step, X_train, y_train)
for accelFile, gyroFile in testDirs:
    userBatches(accelFile, gyroFile,N_TIME_STEPS, step, X_test, y_test)


# In[18]:


print(len(X_train))
X_train


# In[19]:


print(len(y_train))
y_train


# In[20]:


print(len(X_test))
X_test


# In[21]:


print(len(y_test))
y_test


# Convert traning and test data to numpy arrays.

# In[22]:


y_train = np.asarray(y_train, dtype = np.float32)
X_train = np.asarray(X_train, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
X_test = np.asarray(X_test, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
y_test = np.asarray(y_test, dtype = np.float32)


# In[23]:


X_train


# In[134]:


X_test


# In[24]:


y_train


# In[25]:


y_test


# Incorporate Data Augmentation Through Permutations

# In[27]:


aug_x = np.concatenate([X_train, permutation(X_train)])
aug_y = np.concatenate([y_train, y_train])


# # Initialize Hyperparameters

# In[28]:


verbose = 1
epochs = 10
batch_size = 400

n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]
n_outputs = y_train.shape[1]

print('n_timesteps : ', n_timesteps)
print('n_features : ', n_features)
print('n_outputs : ', n_outputs)


# # Model

# Opted for 1d Convolutional layers because they work well with time series data. Added dropout layers and l2 regularizers to reduce some of the overfitting that was occurring during testing.

# In[29]:


model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding = 'same', kernel_regularizer=reg.l2(l=0.15)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding = 'same', kernel_regularizer=reg.l2(l=0.15)))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding = 'same', kernel_regularizer=reg.l2(l=0.15)))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(n_outputs, activation='softmax'))


# In[30]:


model.summary()


# In[33]:


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate= 0.001), metrics=['accuracy', 'mean_absolute_error'])

# fit network
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=epochs, batch_size=batch_size, verbose=verbose)


# evaluate model
(loss, accuracy, mae) = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


# In[ ]:




model.save('models/full_model.h5')
# # Model Metrics

# In[34]:


# confusion matrix
LABELS = ['WALKING',
          'JOGGING',
          'STAIRS',
          'SITTING',
          'STANDING']
y_pred_test = model.predict(X_test,  verbose=0)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(matrix,
            cmap='PuOr',
            linecolor='white',
            linewidths=1,
            xticklabels=LABELS,
            yticklabels=LABELS,
            annot=True,
            fmt='d')
#Most Misclassifications Occur between Stairs and Walking
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[227]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[228]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[229]:


plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Quantize

# In[30]:


MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'model'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'


# In[243]:


def representative_dataset():
   X_rep = train_test_split(X_train, y_train, test_size=0.3, random_state=RANDOM_SEED)[1]
   yield [X_rep]         
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_type = tf.int8
converter.inference_input_type = tf.int8 
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Save the model.
tflite_filename = 'quantized__model.tflite'
with open(tflite_filename, 'wb') as f:
   f.write(tflite_quant_model)


# In[224]:


print("Size of Quantized Model: " + str(len(tflite_quant_model)))


# # Inference

# In[225]:


#Test the accuracy of the quantized model on the test set.
interpreter = tf.lite.Interpreter(model_path = tflite_filename)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

accurate = 0
X_test_int8 = np.asarray(X_test/input_scale + input_zero_point, dtype=np.int8)
for i in range(len(X_test)):
    X_test_int8_sample = np.array([X_test_int8[i]])

    interpreter.set_tensor(input_details[0]['index'], X_test_int8_sample)
    interpreter.invoke()

    outputCategories = np.asarray(interpreter.get_tensor(output_details[0]['index']), dtype=np.float32)
    Categories = (outputCategories - output_zero_point) * output_scale
    if (np.argmax(Categories[0]) == np.argmax(y_test[i])):
        accurate += 1
print(str(accurate/len(y_test) * 100) + " % Quantized Accuracy on the test set")


# In[179]:


#Convert to .h file using xxd
tfmicro_filename = tflite_filename.split('.')[0] + '.h'
get_ipython().system('xxd -i $tflite_filename > $tfmicro_filename')


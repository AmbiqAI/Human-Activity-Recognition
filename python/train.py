import os
import sys
import pydantic_argparse
from params import TrainParams
from utils import save_pkl, load_pkl

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
# from tensorflow.keras.optimizers import optimizers 
import tensorflow_model_optimization as tfmot
import os
from scipy.interpolate import CubicSpline      # for warping
# from transforms3d.axangles import axangle2mat  # for rotation

from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras import regularizers as reg
RANDOM_SEED = 42

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam

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
    print("Orig Steps")
    print(orig_steps)
    print("num_segs")
    print(num_segs)
    for i, pat in enumerate(x):
        print("i = %d" % i)
        print("pat")
        print(pat)
       
        if num_segs[i] > 1:
            splits = np.array_split(orig_steps, num_segs[i])
            print("splits")
            print(splits)
            print("perm")
            print(np.random.permutation(splits))
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            print("warp")
            print(warp)
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret



def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    X_new[:,3] = np.interp(x_range, tt_new[:,3], X[:,3])
    X_new[:,4] = np.interp(x_range, tt_new[:,4], X[:,4])
    X_new[:,6] = np.interp(x_range, tt_new[:,5], X[:,5])
    return X_new

def augment(X, jitter_sigma=0.05, scaling_sigma=0.1):
    X_new = np.zeros(X.shape)
    # print(X)
    for i,orig in enumerate(X):
        # print(i)
        # print(orig)
        jitterNoise = np.random.normal(loc=0, scale=jitter_sigma, size=orig.shape)
        scaleNoise = np.random.normal(loc=1.0, scale=scaling_sigma, size=orig.shape)
        X_new[i] = orig*scaleNoise+jitterNoise
        # X_new[i] = DA_TimeWarp(X_new[i])

        # if i==0:
        #     fig = plt.figure(figsize=(15,4))
        #     oplt = fig.add_subplot(2,4,1)
        #     oplt.plot(orig)
        #     xplt = fig.add_subplot(2,4,2)
        #     xplt.plot(X_new[i])  
        #     plt.show()  
    return X_new


def userBatches(accelFile, gyroFile, windowSize, stride, windows, vectorizedActivities):
    accel = pd.read_csv(accelFile, names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    gyro = pd.read_csv(gyroFile, names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    accel['Z'] = accel['Z'].str[0: -1]
    gyro['Z'] = gyro['Z'].str[0: -1]
    # print (accel)
    # print (accel.index)
    # if accel.loc[0,:]['user id'] != gyro.loc['0.0',:]['user id']:
    #     print("mismatch")

    for label in labels:
        #merge accel and gyroscope data to have 6 channels
        filt1 = accel.where(accel['Activity Label'] == label).dropna().sort_values('Time Stamp').iloc[:, [3,4,5]]
        filt2 = gyro.where(gyro['Activity Label'] == label).dropna().sort_values('Time Stamp').iloc[:, [0,3,4,5]]
        merged = filt1.reset_index().merge(filt2.reset_index(), left_index=True, right_index=True, how='left').dropna()
        merged = merged.drop(['index_x','index_y', 'user id'], axis=1)
        # print (merged.shape[0])
        for i in range(0, merged.shape[0] - windowSize, stride):
            newWindow = merged.iloc[i:i+windowSize, :].astype(float)
            normalize_window(newWindow)
            newWindow = newWindow.to_numpy().tolist()
            ctgrs = lb.transform([label]).tolist()[0]
            vectorizedActivities.append(ctgrs)
            windows.append(newWindow)

def representative_dataset():
   X_rep = train_test_split(X_train, y_train, test_size=0.3, random_state=RANDOM_SEED)[1]
   yield [X_rep]         

# def train_model(params: TrainParams):


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=TrainParams,
        prog="Human Activity Recognition Train Command",
        description="Train HAR model",
    )

if __name__ == "__main__":
    # parser = create_parser()
    # train_model(parser.parse_typed_args())

    accelDirs = list(os.scandir('accel'))[2:53]
    print(accelDirs)
    del accelDirs[37:41]
    gyroDirs = list(os.scandir('gyro'))[2:53]
    del gyroDirs[37:41]
    labels = ['A', 'B', 'C', 'D', 'E']
    allDirs = list(zip(accelDirs, gyroDirs))
    trainingIndices = np.random.choice(len(allDirs), 35, replace=False).tolist() # TODO make 35 not literal
    trainingDirs = [allDirs[i] for i in range(len(allDirs)) if i in trainingIndices]

    print("Number of users in Training Set: %d" % len(trainingIndices))
    trainingIndices.sort()
    # for index in trainingIndices:
    #     print("User " + str(index))
    #rest go to test set
    testDirs = [x for x in allDirs if x not in trainingDirs]
    # print("Users in Test Set:\n")
    # for index in [x for x in range(len(allDirs)) if x not in trainingIndices]:
    #     print("User " + str(index))
    
    lb = LabelBinarizer()
    print(labels)
    print("Before lb")    
    lb.fit(labels)
    #EXAMPLE OF OUTPUT
    lb.transform(['A'])
    print("after transform lb")    
    print("After lb")    

    # Creates 200 timestep windows that slide 20 timesteps at each iteration
    N_TIME_STEPS = 200
    N_FEATURES = 6
    step = 20

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #CREATING BATCHES OF WINDOWS FOR EACH USER IN TEST & TRAIN SET
    # for accelFile, gyroFile in trainingDirs:
    #     userBatches(accelFile, gyroFile, N_TIME_STEPS, step, X_train, y_train)
    # for accelFile, gyroFile in testDirs:
    #     userBatches(accelFile, gyroFile,N_TIME_STEPS, step, X_test, y_test)

    # save_pkl("test.pkl", X=X_train, y=y_train, XT=X_test, yt=y_test)
    # ds = load_pkl("test.pkl")
    # X_train = ds["X"]
    # y_train = ds["y"]
    # X_test = ds["XT"]
    # y_test = ds["yt"]

    # print(len(X_train))
    # print(len(y_train))
    
    # print(len(X_test))
    # print(len(y_test))

    # # Convert traning and test data to numpy arrays.

    # y_train = np.asarray(y_train, dtype = np.float32)
    # X_train = np.asarray(X_train, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    # X_test = np.asarray(X_test, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    # y_test = np.asarray(y_test, dtype = np.float32)
    # save_pkl("nparray.pkl", X=X_train, y=y_train, XT=X_test, yt=y_test)
    
    ds = load_pkl("nparray.pkl")
    X_train = ds["X"]
    y_train = ds["y"]
    X_test = ds["XT"]
    y_test = ds["yt"]
    # Incorporate Data Augmentation Through Permutations
    print("Before Augmentation X_train len is %d "%+ len(X_train))

    aug_x = np.concatenate([X_train, augment(X_train)])
    aug_x = np.concatenate([aug_x, augment(X_train)])
    aug_x = np.concatenate([aug_x, augment(X_train)])
    aug_x = np.concatenate([aug_x, augment(X_train)])
    # aug_x = np.concatenate([X_train, DA_Jitter(DA_TimeWarp(X_train))])

    print("After Augmentation X_train len is %d " % len(aug_x))
    aug_y = np.concatenate([y_train, y_train])
    aug_y = np.concatenate([aug_y, y_train])
    aug_y = np.concatenate([aug_y, y_train])
    aug_y = np.concatenate([aug_y, y_train])

    # # Initialize Hyperparameters
    verbose = 1
    epochs = 20
    batch_size = 400

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    print('n_timesteps : ', n_timesteps)
    print('n_features : ', n_features)
    print('n_outputs : ', n_outputs)

    # # Model
    # Opted for 1d Convolutional layers because they work well with time series data. Added dropout layers and l2 regularizers to reduce some of the overfitting that was occurring during testing.

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

    model.summary()


    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate= 0.001), metrics=['accuracy', 'mean_absolute_error'])

    # fit network
    history = model.fit(aug_x, aug_y, validation_data=(X_test, y_test), 
                        epochs=epochs, batch_size=batch_size, verbose=verbose)


    # evaluate model
    (loss, accuracy, mae) = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


    model.save('models/full_model.h5')
    # # Model Metrics

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

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    # # Quantize

    MODELS_DIR = 'trained_models/'
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    MODEL_TF = MODELS_DIR + 'model'
    MODEL_TFLITE = MODELS_DIR + 'model.tflite'
    MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

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
    print("Size of Quantized Model: " + str(len(tflite_quant_model)))
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
    tfmicro_filename = tflite_filename.split('.')[0] + '.h'
    # get_ipython().system('xxd -i $tflite_filename > $tfmicro_filename')
import os
import sys
import pydantic_argparse
from params import TrainParams
from utils import save_pkl, load_pkl, xxd_c_dump

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
    print("len of x_new is %d" % len(X_new))
    return X_new


def userBatches(accelFile, gyroFile, windowSize, stride, windows, vectorizedActivities, lb, labels):
    print(accelFile)
    accel = pd.read_csv(accelFile, names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    gyro = pd.read_csv(gyroFile, names=['user id', 'Activity Label', 'Time Stamp', 'X', 'Y', 'Z'])
    accel['Z'] = accel['Z'].str[0: -1]
    gyro['Z'] = gyro['Z'].str[0: -1]

    for label in labels:
        #merge accel and gyroscope data to have 6 channels
        filt1 = accel.where(accel['Activity Label'].str.strip() == label).dropna().sort_values('Time Stamp').iloc[:, [3,4,5]]
        filt2 = gyro.where(gyro['Activity Label'].str.strip() == label).dropna().sort_values('Time Stamp').iloc[:, [0,3,4,5]]
        merged = filt1.reset_index().merge(filt2.reset_index(), left_index=True, right_index=True, how='left').dropna()
        merged = merged.drop(['index_x','index_y', 'user id'], axis=1)

        for i in range(0, merged.shape[0] - windowSize, stride):
            newWindow = merged.iloc[i:i+windowSize, :].astype(float)
            normalize_window(newWindow)
            newWindow = newWindow.to_numpy().tolist()
            ctgrs = lb.transform([label]).tolist()[0]
            vectorizedActivities.append(ctgrs)
            windows.append(newWindow)
        print(".", end = "")


def get_dataset(params: TrainParams, fine_tune = False):
    # If augmented dataset pkt exists, load that
    #  otherwise, if pre-processed baseline dataset exists, load that and augment it
    #   otherwise, load the raw data, process it, and augment it

    if fine_tune:
        aug_file = params.dataset_dir+params.augmented_ft_dataset
    else:
        aug_file = params.dataset_dir+params.augmented_dataset

    if aug_file and os.path.isfile(aug_file):
        print("Loading augmented dataset from " + aug_file)
        ds = load_pkl(aug_file)
        return ds["X"], ds["y"], ds["XT"], ds["yt"]
    
    if fine_tune:
        processed_file = params.dataset_dir+params.processed_ft_dataset
    else:
        processed_file = params.dataset_dir+params.processed_dataset
    
    if not os.path.isfile(processed_file):
        # Generate a processed file from scratch
        print("Creating processed dataset")

        if fine_tune:
            accelDirs = list(os.scandir(params.dataset_dir+'/accel_finetune'))[2:53]
            gyroDirs = list(os.scandir(params.dataset_dir+'/gyro_finetune'))[2:53]
        else:
            accelDirs = list(os.scandir(params.dataset_dir+'/accel'))[2:53]
            gyroDirs = list(os.scandir(params.dataset_dir+'/gyro'))[2:53]

        del accelDirs[37:41]
        del gyroDirs[37:41]
        allDirs = list(zip(accelDirs, gyroDirs))
        training_files_count = round(len(accelDirs) * params.training_dataset_percent / 100)
        # print(training_files_count)
        trainingIndices = np.random.choice(len(allDirs), training_files_count, replace=False).tolist()
        trainingDirs = [allDirs[i] for i in range(len(allDirs)) if i in trainingIndices]     
        testDirs = [x for x in allDirs if x not in trainingDirs]

        labels = ['A', 'B', 'C', 'D', 'E']
        lb = LabelBinarizer()  
        lb.fit(labels)

        aug_X = []
        aug_y = []
        testX = []
        testy = []

        # Create batches by moving a sampling window over each user's data
        for accelFile, gyroFile in trainingDirs:
            userBatches(accelFile, gyroFile, params.num_time_steps, params.sample_step, aug_X, aug_y, lb, labels)
            print("-", end = "")
        for accelFile, gyroFile in testDirs:
            userBatches(accelFile, gyroFile, params.num_time_steps, params.sample_step, testX, testy, lb, labels)
            print("-", end = "")
        print("")
        aug_y = np.asarray(aug_y, dtype = np.float32)
        aug_X = np.asarray(aug_X, dtype= np.float32).reshape(-1, params.num_time_steps, params.num_features)
        testX = np.asarray(testX, dtype= np.float32).reshape(-1, params.num_time_steps, params.num_features)
        testy = np.asarray(testy, dtype = np.float32)   

        save_pkl(processed_file, X=aug_X, y=aug_y, XT=testX, yt=testy)
        if params.augmentations == 0:
            return aug_X, aug_y, testX, testy

    else:
        # Load it
        print("Loading processed dataset from " + processed_file)
        ds = load_pkl(processed_file)
        aug_X = ds["X"]
        aug_y = ds["y"]
        testX = ds["XT"]
        testy = ds["yt"]
        if params.augmentations == 0:
            return aug_X, aug_y, testX, testy
    
    # Augment if any are requested
    print("Augmenting baseline by %dx" % params.augmentations)
    orig_X = aug_X
    orig_testX = testX
    orig_y = aug_y
    orig_testy = testy
    
    for i in range(params.augmentations):
            aug_X = np.concatenate([aug_X, augment(orig_X)])       
            aug_y = np.concatenate([aug_y, orig_y])       

    if params.save_processed_dataset:
        save_pkl(aug_file, X=aug_X, y=aug_y, XT=testX, yt=testy)

    return aug_X, aug_y, testX, testy


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

def decay(epoch):
        if epoch < 15:
            return 1e-3
        if epoch < 30:
            return 1e-4
        return 1e-5

def plot_training_results(model, history):

    # Model Metrics
    # confusion matrix
    LABELS = ['WALKING',
            'JOGGING',
            'STAIRS',
            'SITTING',
            'STANDING']
    y_pred_test = model.predict(test_data,  verbose=0)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(test_labels, axis=1)

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


def load_existing_model(params: TrainParams):
    return load_model(params.trained_model_dir + "/" + params.model_name + ".h5") 


def train_model(params: TrainParams, train_data, train_labels, test_data, test_labels, fine_tune = False):
    # Initialize Hyperparameters
    verbose = 1

    if fine_tune:
        epochs = params.ft_epochs
    else:
        epochs = params.epochs

    batch_size = params.batch_size

    n_timesteps = aug_data.shape[1]
    n_features = aug_data.shape[2]
    n_outputs = aug_labels.shape[1]

    print('[INFO] n_timesteps : ', n_timesteps)
    print('[INFO] n_features : ', n_features)
    print('[INFO] n_outputs : ', n_outputs)

    # Model Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="accuracy",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )

    checkpoint_weight_path = str(params.job_dir) + "/model.weights"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_weight_path,
        monitor="accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        verbose=1,
    )
    tf_logger = tf.keras.callbacks.CSVLogger(str(params.job_dir) + "/history.csv")
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay)
    model_callbacks = [early_stopping, checkpoint, tf_logger, lr_scheduler]

    # Model
    if fine_tune:
        model = load_existing_model(params)
    else:
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

        model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
            metrics=['accuracy', 'mean_absolute_error'])
    
    model.summary()

    # fit network
    history = model.fit(aug_data, aug_labels, validation_data=(test_data, test_labels), 
                        epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=model_callbacks,)

    # evaluate model
    (loss, accuracy, mae) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=verbose)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%, mean absolute error={:..4f}%".format(loss, accuracy * 100, mae))
    
    model.save(params.trained_model_dir + "/" + params.model_name + ".h5")

    return model, history


if __name__ == "__main__":
    parser = create_parser()
    params = parser.parse_typed_args()

    # Load Baseline Data
    aug_data, aug_labels, test_data, test_labels = get_dataset(params, False)

    # Load Fine-tune Data
    ft_aug_data, ft_aug_labels, ft_test_data, ft_test_labels = get_dataset(params, True)
    
    # Train model
    if params.train_model:
        model, history = train_model(params, aug_data, aug_labels, test_data, test_labels, fine_tune=False)
        if params.show_training_plot:
            plot_training_results(model, history)
    else:
        model = load_existing_model(params)
        model.summary()

    # Fine-tune model
    if params.fine_tune_model:
        model, history = train_model(params, ft_aug_data, ft_aug_labels, ft_test_data, ft_test_labels, fine_tune=True)
        if params.show_training_plot:
            plot_training_results(model, history)

    # Quantize and convert
    # MODELS_DIR = 'trained_models/'
    # if not os.path.exists(MODELS_DIR):
    #     os.mkdir(MODELS_DIR)
    # MODEL_TF = MODELS_DIR + 'model'
    # MODEL_TFLITE = MODELS_DIR + 'model.tflite'
    # MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

    tflite_filename = params.trained_model_dir + "/" + params.model_name + ".tflite"
    tflm_filename = params.trained_model_dir + "/" + params.model_name + ".cc"
    
    def representative_dataset():
        # X_rep = train_test_split(data, label, test_size=0.3, random_state=seed)[1]
        yield [test_data]    # TODO get better dataset, but aug_data is too large

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
    print("[INFO] Size of Quantized Model: " + str(len(tflite_quant_model)))
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_quant_model)
    
    # Evaluate tflite model
    interpreter = tf.lite.Interpreter(model_path = tflite_filename)
    interpreter.allocate_tensors() 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    accurate = 0
    X_test_int8 = np.asarray(test_data/input_scale + input_zero_point, dtype=np.int8)
    for i in range(len(test_data)):
        X_test_int8_sample = np.array([X_test_int8[i]])

        interpreter.set_tensor(input_details[0]['index'], X_test_int8_sample)
        interpreter.invoke()

        outputCategories = np.asarray(interpreter.get_tensor(output_details[0]['index']), dtype=np.float32)
        Categories = (outputCategories - output_zero_point) * output_scale
        if (np.argmax(Categories[0]) == np.argmax(test_labels[i])):
            accurate += 1
    print(str(accurate/len(test_labels) * 100) + " % Quantized Accuracy on the test set")
    # tfmicro_filename = tflite_filename.split('.')[0] + '.h'
    xxd_c_dump(
        src_path=tflite_filename,
        dst_path=tflm_filename,
        var_name='har_model',
        chunk_len=12,
        is_header=True,
    )
    # os.system('xxd -i $tflite_filename > $tflm_filename')
# Dataset

Instead of gathering live sensor readings, we relied on a public dataset [dataset](https://www.cis.fordham.edu/wisdm/dataset.php).

The WISDM lab recorded this data through controlled experiements, where they made 50 different
individuals (each subject has their own file in data/accel and data/gyro) perform 18 different tasks.
Each task was performed for 3 mins by each user, yielding around 3600 readings per user per activity
(20 sensor readings/second). We only trained the model in predicting 5 of the 18 activites: Walking,
Laying, Sitting, Standing, and Going up the Stairs. We excluded activities like eating chips and folding clothes from the model because we do not believe they can be reasonably differentiated throught the use of gyroscope and accelerometer data.


# Approach

The training procedure takes inspiration from another HAR model that used a similar dataset[authors]
(https://github.com/shafiqulislamsumon/HumanActivityRecognition). However, there were several drastic
changes:

1. Making sure that time series windows were specific to 1 user and 1 activity.
2. Incorporating gyroscope readings into the model.
3. Seperating user data files used in training from those used in validation.
4. Adding l2 regularizers and dropout layers
5. Tweaking the hyperparameters of the CNN so that each successive layer had more filters than the previous. 
6. Introducing permutations of the time series windows as a means of data augmentation.

Overall, the main inspiration of the linked repo was the use of sliding windows in structuring the training set. The afformentioned changes were implemented to curb some of the overfitting that we suspected of the original example and to match the specifics of our dataset.

# Sensor/Data Collection

This example relies on the MPU6050 gyrometer. The sensor is set to have a accelerometer sensitivity of 16G and a gyroscope sensitivity of 500DPS. Although normalizing each window helps account for the differences between the sensors used in the WISDM experiment and the MPU6050, we found it necessary to tune the model. This was done in 'Model Tuning.ipynb', where we utilized an hour's worth of our own activity data (stored in mpu-6050-data.txt) to further train the model.

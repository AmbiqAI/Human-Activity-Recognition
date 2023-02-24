import os
import sys
import pydantic_argparse
from params import TrainParams

import numpy as np
import pandas as pd

# def convert_mpu_data(filename):
    # Load captured data file, which has accel and gyro for several users and activities
mpuData = pd.read_csv('datasets/mpu-6050-data.txt')
# mpuData = mpuData.drop(['Unnamed: 0'], axis=1)
actMap = {
    'Walking' : 'A',
    'Jogging': 'B',
    'Stairs': 'C',
    'Sitting': 'D',
    'Standing': 'E'
}
print (mpuData['Activity'])
# Create accel and gyro dataframes
#accelData = mpuData.where(accel['Activity'] == label).dropna().sort_values('Time Stamp').iloc[:, [3,4,5]]
mpuData = mpuData.replace({'Activity': actMap})
accelData = mpuData.iloc[:, [1,8, 2,3,4]]
gyroData = mpuData.iloc[:, [1,8, 5,6,7]]
print (accelData)
print (gyroData)
for user in [*set(accelData["User"])]:
    accelFile = "data_"+str(user)+"_accel_watch.txt"
    print (accelFile)
    usersData = mpuData.where(mpuData["User"] == user).dropna()
    with open(accelFile,'w') as f:
        for i in range(len(usersData)):
            row = usersData.iloc[i]
            # user, activity, timestamp, x, y, z
            print ("%d, %s, %d, %f, %f, %f;" % 
                (row["User"], row["Activity"], row["timestamp"], 
                row['accelX'], row['accelY'], row['accelZ']), file=f
            )

    gyroFile = "data_"+str(user)+"_gyro_watch.txt"
    print (gyroFile)
    usersData = mpuData.where(mpuData["User"] == user).dropna()
    with open(gyroFile,'w') as f:
        for i in range(len(usersData)):
            row = usersData.iloc[i]
            # user, activity, timestamp, x, y, z
            print ("%d, %s, %d, %f, %f, %f;" % 
                (row["User"], row["Activity"], row["timestamp"], 
                row['gyroX'], row['gyroY'], row['gyroZ']), file=f
            )

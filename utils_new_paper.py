"""
Module providing functions to read datasets
"""

import os, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ahrs

from project_constants import samp_rate, region_map, region_map_merge



def _read_file(filename):
    """
    Each file should have the same structure
    epoch, timestamp and elapsed columns reprenting sample time and 3 data channels
    """

    # read raw file
    df=pd.read_csv(filename)

    # drop unused columns
    df = df.drop(["epoc (ms)","elapsed (s)"],axis=1)

    # convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp (+1000)"], format='%Y-%m-%dT%H.%M.%S.%f')

    df = df.drop(["timestamp (+1000)"],axis=1)

    # index on timestamp
    df = df.set_index("timestamp")

    return df



def _interpolate(df, new_index, limit=1): # 5 milliseconds  
    return df.reindex(df.index.union(new_index)).fillna(method='ffill', limit=limit).loc[new_index]

    

def read_sensor_data(
    path:str,                  # location of experiments
    experiment:str,            # experiment folder
    device:str,                # name of device (W1, W2)):
    freq="5L",                 # "5L" = 5ms = 200Hz
    ) ->  pd.DataFrame:         
    """ loads raw sensor data into dataframe """
    experiment_path = os.path.join(path, experiment)
    accelerometer_filename = glob.glob(os.path.join(experiment_path, f"*{device}*-A.csv"))[0]
    gyroscope_filename = glob.glob(os.path.join(experiment_path, f"*{device}*-G.csv"))[0]

    # read the three sensors
    accelerometer = _read_file(accelerometer_filename)
    gyroscope = _read_file(gyroscope_filename)

    # in order to join we need to resample and interpolate
    start = np.max([
        accelerometer.index.min().ceil(freq), 
        gyroscope.index.min().ceil(freq), 
        ])
    end = np.min([
        accelerometer.index.max().floor(freq), 
        gyroscope.index.max().floor(freq), 
        ])
    new_index = pd.date_range(start, end, freq=freq)

    accelerometer = _interpolate(accelerometer, new_index)
    gyroscope = _interpolate(gyroscope, new_index)

    df = pd.concat((accelerometer, gyroscope), axis=1)
    return df


def load_dataset_new_paper(
    path,
    experiment, 
    eval_config, 
    timezone="Australia/Sydney",
    freq="5L"
    ):
    # load data

    device = eval_config['device']

    df = read_sensor_data(path=path,experiment=experiment,device=device,freq=freq)
    df.index = df.index.tz_localize(timezone)
    df.index.name = 'timestamp'

    # load labels
    filename = os.path.join(path, experiment, f'labels.json')
    with open(filename, 'r') as fp:
        labels = json.load(fp)

    labels = pd.DataFrame.from_dict(labels, orient='index', columns=["from","to"])
    labels["from"] = pd.to_datetime(labels["from"]).dt.tz_convert(timezone) # have to convert from fixed offset
    labels["to"] = pd.to_datetime(labels["to"]).dt.tz_convert(timezone)
    labels = labels.sort_values("from")

    # locate the activities
    start = labels["from"].min()
    end = labels["to"].max()
    df = df[start:end]

    # label the activities
    start_idx = np.searchsorted(labels["from"].values, df.index.values)-1
    end_idx = np.searchsorted(labels["to"].values, df.index.values)
    mask = (start_idx == end_idx)

    df.loc[:, 'label']=None
    df.loc[mask,'label']=labels.index[start_idx[mask]]
    df.loc[:, "label"]=df.loc[:, "label"].bfill() # hack mainly to fix first row missing label

    df.dropna(axis=0, inplace=True)

    df.loc[:, "angle__x-axis (deg)"] = df.loc[:, "x-axis (deg/s)"].cumsum() / 200.0
    df.loc[:, "angle__y-axis (deg)"] = df.loc[:, "y-axis (deg/s)"].cumsum() / 200.0
    df.loc[:, "angle__z-axis (deg)"] = df.loc[:, "z-axis (deg/s)"].cumsum() / 200.0
    

    gyro = df.loc[:, ["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"]].values
    accel = df.loc[:, ["x-axis (g)", "y-axis (g)", "z-axis (g)"]].values

    if eval_config['use_euler_angles']:
        #euler angles form ahrs madgwick
        ahrsEst = ahrs.filters.Madgwick(acc=accel*9.8, gyr=gyro*(np.pi/180), frequency = samp_rate)
        
        # euler angles from ahrs complimentary filter
        # ahrsEst = ahrs.filters.complementary.Complementary(acc=accel*9.8, gyr=gyro*(np.pi/180), frequency = samp_rate, gain=0.1)
        
        quat = ahrs.QuaternionArray(ahrsEst.Q) 
        euler_angles = quat.to_angles()
        EA_df = pd.DataFrame(data=euler_angles, columns=['roll', 'pitch', 'yaw'])

        # euler angles from paper's complimentary filter
        # euler_angles = complimentary_filt(accel, gyro)
        # EA_df = pd.DataFrame(data=euler_angles, columns=['roll', 'pitch'])

        EA_df.index = df.index
        df = pd.concat((df,EA_df), axis=1)


    df.dropna(axis=0, inplace=True)

    num_dental_regs = eval_config['num_dental_regs']
    region_map_selected = region_map
    if num_dental_regs != 16:
        region_map_selected = region_map_merge
    
    labels_sess = df.loc[:, "label"].map(region_map_selected).values

    # df.drop("label", axis=1, inplace=True)

    features_sess = df.loc[:, eval_config['data_fields']].values #, "yaw"  "angle__x-axis (deg)"
    
    if eval_config['use_euler_angles'] == True:
      features_sess = np.hstack([features_sess, euler_angles])


    return features_sess, labels_sess, df



def complimentary_filt(accel, gyro):
    c=0.95 # filter time constant - usually 0.95-0.98

    # compute angles from gyroscope
    gyro = np.radians(gyro)
    angles = np.cumsum(gyro, axis=0) / samp_rate

    # remove gravity from accel - low pass filter to (approximately) separate linear acceleration from gravity
    g = accel[0,:]
    alpha=0.8
    gravity=np.zeros(angles.shape)
    for i in range(len(angles)):
        g = 0.8 * g + (1-alpha) * accel[i,:]
        gravity[i,:] = g

    accel = accel - g

    # apply complimentary filter - combines integrated gyro pitch / roll and same angles calculated from linear accelerations
    attitudes=np.zeros((angles.shape[0],2))
    for i in range(len(attitudes)):
        # https://www.nxp.com/files-static/sensors/doc/app_note/AN3461.pdf
        roll=angles[i,0]
        accel_roll = np.arctan2(accel[i,1], accel[i,2])
        roll=c*roll+(1-c)*accel_roll

        pitch=angles[i,1]
        accel_pitch = np.arctan2(-accel[i,0], np.sqrt(accel[i,1]*accel[i,1] + accel[i,2]*accel[i,2]))
        pitch=c*pitch+(1-c)*accel_pitch

        attitudes[i,:] = np.array([pitch,roll])
    
    # df['pitch'] = attitudes[:,0]
    # df['roll'] = attitudes[:,1]

    # return np.hstack(attitudes[:,1], attitudes[:,0], attitudes[:,3])
    return attitudes
    

    # plt.plot(attitudes)
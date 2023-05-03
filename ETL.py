import os
from operator import itemgetter, truediv
from collections import defaultdict
from itertools import groupby
import pickle
from xml.sax.handler import feature_string_interning
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset
import ahrs
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_class import RobasDataset
from global_constants import region_map, region_map_merge, days_not_labeled, days_poor_acc, left_handed_patients, samp_freq
from project_constants import unlabeled_folders, patient_out_not_include, session_out_not_include
from utils_new_paper import load_dataset_new_paper

def sample_based_data_transform(dataloader):
    x_data = dataloader.dataset[0][0]; y_data = np.repeat(dataloader.dataset[0][1], x_data.shape[0]) # y_data = dataloader.dataset[0][1] 
    skip_first = 0
    for batch_data, batch_labels, batch_brush_types, batch_is_left_handed in dataloader.dataset: #first batch gets in there twice btw
        skip_first += 1
        if skip_first == 1:
            continue
        x_data = np.vstack([x_data, batch_data])
        y_data = np.append(y_data, np.repeat(batch_labels, batch_data.shape[0]))
        # y_data = np.append(y_data, batch_labels)

    return x_data, y_data

def create_all_datasets(data_path, files_meta_config, eval_config, segment_config, seed=0, shuffle=True):
    data_files_names = os.listdir(data_path)
    count_unfiltered_files = 0
    all_dataset = []
    all_labels = []
    all_files = []


    # counter = 0
    for file_name in data_files_names:
        curr_file_meta_config = files_meta_config[files_meta_config["file_name"]==file_name].copy()
        if filter_files(file_name, curr_file_meta_config, eval_config):
            continue
        
        count_unfiltered_files += 1

        file_dataset, features_sess, labels_sess = file_to_dataset(data_path, file_name, curr_file_meta_config, eval_config, segment_config)
        all_dataset.extend(file_dataset) # = ConcatDataset([all_dataset, file_dataset]) #would it append none if there is no segment for a specific regions?

        all_labels.extend(labels_sess)
        all_files.extend([file_name]*len(labels_sess))
        # counter += 1
        # if counter >= 7:
        #     break

 
    nan_indices = np.nonzero(np.isnan(all_labels))[0]
    # all_features_new = np.delete(all_features, nan_indices, axis=0)
    # all_labels_new = np.delete(all_labels, nan_indices, axis=0)

    if len(nan_indices) > 0:
        print("nan indices exist go and check")
        print(f"{set(all_files[nan_indices])}")


    all_dataset_concat = ConcatDataset(all_dataset)

    if shuffle:
        np.random.seed(seed)
        random_indices = list(np.random.choice(len(all_dataset_concat), len(all_dataset_concat)))
        all_dataset_concat = list(itemgetter(*random_indices)(all_dataset_concat))
    

    return all_dataset_concat


def create_patients_datasets_dict(data_path, files_meta_config, eval_config, segment_config, seed=0, shuffle=True):
    patient_files = defaultdict(list)
    patient_dataset_dict = defaultdict(list)
    data_files_names = os.listdir(data_path)
    count_unfiltered_files = 0

    # counter = 0
    for file_name in data_files_names:
        curr_file_meta_config = files_meta_config[files_meta_config["file_name"]==file_name].copy()
        if filter_files(file_name, curr_file_meta_config, eval_config):
            continue
        
        count_unfiltered_files += 1

        file_dataset, _, _ = file_to_dataset(data_path, file_name, curr_file_meta_config, eval_config, segment_config)
        
        patient_id = curr_file_meta_config['patient_id'].values[0]
        # patient_id = file_name.split('.')[0].split("Day")[0]
        patient_dataset_dict[patient_id].extend(file_dataset) # ConcatDataset([patient_dataset_dict[patient_id], file_dataset]) #would it append none if there is no segment for a specific regions?
        patient_files[patient_id].append(file_name)

        # counter += 1
        # if counter >= 7:
        #     break

    with open(os.path.join(os.path.dirname(__file__), f"patient_files_dict_{eval_config['dataset_type']}"), "wb") as data_file:  # Pickling
        pickle.dump(patient_files, data_file)
        

    for patient_id, patient_dataset in patient_dataset_dict.items():

        patient_dataset_concat = ConcatDataset(patient_dataset)

        if shuffle:
            np.random.seed(seed)
            random_indices = list(np.random.choice(len(patient_dataset_concat), len(patient_dataset_concat)))
            patient_dataset_concat = list(itemgetter(*random_indices)(patient_dataset_concat))
    
        patient_dataset_dict[patient_id] = patient_dataset_concat

    return patient_dataset_dict

def create_sessions_datasets_dict(data_path, files_meta_config, eval_config, segment_config, seed=0, shuffle=True):
    patient_files = defaultdict(list)
    session_dataset_dict = defaultdict(list)
    data_files_names = os.listdir(data_path)
    count_unfiltered_files = 0
    # counter = 0
    for file_name in data_files_names:
        curr_file_meta_config = files_meta_config[files_meta_config["file_name"]==file_name].copy()
        if filter_files(file_name, curr_file_meta_config, eval_config):
            continue
        
        count_unfiltered_files += 1

        file_dataset, _, _ = file_to_dataset(data_path, file_name, curr_file_meta_config, eval_config, segment_config)
        patient_id, session_id = curr_file_meta_config['patient_id'].values[0], int(curr_file_meta_config['session_id'].values[0]) #.split('Day')[1]
        # patient_id, session_id = file_name.split('.')[0].split("Day")
        
        if session_dataset_dict[patient_id] == []:
            session_dataset_dict[patient_id] = defaultdict(list)
        
        session_dataset_dict[patient_id][session_id] = file_dataset #would it append none if there is no segment for a specific regions?
        patient_files[patient_id].append(file_name)

        # counter += 1
        # if counter >= 7:
        #     break

    with open(os.path.join(os.path.dirname(__file__), f"patient_files_dict_{eval_config['dataset_type']}"), "wb") as data_file:  # Pickling
        pickle.dump(patient_files, data_file)
        
    for patient_id, patient_dataset_dict in session_dataset_dict.items():

        for session_id, session_dataset in patient_dataset_dict.items():

            session_dataset_concat = ConcatDataset(session_dataset)

            if shuffle:
                np.random.seed(seed)
                random_indices = list(np.random.choice(len(session_dataset_concat), len(session_dataset_concat)))
                session_dataset_concat = list(itemgetter(*random_indices)(session_dataset_concat))
        
            session_dataset_dict[patient_id][session_id] = session_dataset_concat

    return session_dataset_dict

def file_to_dataset(data_path, file_name, curr_file_meta_config, eval_config, segment_config):
    file_dataset = []

    if eval_config['dataset_type'] == 'mine':
        features_sess, labels_sess, _ = load_dataset_mine(data_path, file_name, eval_config, mode='train')
    elif eval_config['dataset_type'] == 'new_paper':
        features_sess, labels_sess, _ = load_dataset_new_paper(data_path, file_name, eval_config)

    patient_id, session_id = curr_file_meta_config['patient_id'].values[0], int(curr_file_meta_config['session_id'].values[0])
    # patient_id, session_id = file_name.split('.')[0].split("Day")[0], int(file_name.split('.')[0].split("Day")[1])
    
    for label in np.unique(labels_sess[~np.isnan(labels_sess)]):
        index = np.where(labels_sess == label)[0]
        segments_ranges = group_intersection(index)
        for seg_range in segments_ranges:

            dataset = RobasDataset(features = features_sess[seg_range[0]:seg_range[1]+1,:], 
                                    label = label, 
                                    patient_id = patient_id,
                                    session_id = session_id, 
                                    file_meta_config = curr_file_meta_config,
                                    segment_length = segment_config['segment_length'],
                                    segment_stride = segment_config['segment_stride']
                                    )
            
            file_dataset.append(dataset) # = ConcatDataset([file_dataset, dataset])
            
    return file_dataset, features_sess, labels_sess

  
def group_intersection(data):
    ranges =[]
    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
    return ranges

def shift_psi_angle(file_content, file_labels, file_name):
    index_ManLO = np.where(file_labels == region_map['ManLO'])[0]
    

    if any(index_ManLO):
        ManLO_ranges = group_intersection(index_ManLO)
        ManLO_first_range = ManLO_ranges[0]

        diff_of_ManLO_psi = np.pi/6 - file_content[ManLO_first_range[0]:ManLO_first_range[1],-1] # we have to still make them lie in [-pi, pi] after this transformation
        file_content[:,-1] += np.mean(diff_of_ManLO_psi)
    else:
        print(f"\n {file_name} does not have 'ManLO' \n")
    
    return file_content


def process_meta_files(data_path,meta_data_path,dataset_type):
    
    def extract_files_meta_config(patients_df, file_name):
        patient_id, session_id = file_name.split('.')[0].split('Day')[0], int(file_name.split('.')[0].split('Day')[1])
        patient_df = patients_df[patient_id]
        file_meta_config_df = patient_df[patient_df['Day'] == \
            session_id][['Day','Direction', 'Brush', 'Method', 'Head Movement', 'is_left_handed']].copy()
        file_meta_config_df['file_name'] = file_name
        file_meta_config_df['patient_id'] = patient_id
        file_meta_config_df['session_id'] = session_id
        return file_meta_config_df
    
    
    files_meta_config = []

    if dataset_type == 'mine':
        meta_file_df = pd.read_excel(os.path.join(meta_data_path, 'stats.xlsx'), index_col=None, sheet_name=None)     

        patients_df = {}
        for patient_sheet_name, patient_sheet_value in meta_file_df.items():
            patient_num = patient_sheet_name.split(' - ')[0]
            if patient_num in left_handed_patients:
                patient_sheet_value["is_left_handed"] = True
            else:
                patient_sheet_value["is_left_handed"] = False
            patients_df[patient_num] = patient_sheet_value.head(40).copy()

        data_files_names = os.listdir(data_path)
        for file_name in data_files_names:
            if file_name.startswith('.'):
                continue
            files_meta_config.append(extract_files_meta_config(patients_df, file_name))


    elif dataset_type == 'new_paper':
        
        # file_meta_config_df =  pd.DataFrame()

        # for experiment in sorted(glob.glob(os.path.join(PATH, "*"))):
    
        for experiment in os.listdir(data_path):
            folder_name = os.path.join(data_path, experiment)
            if not os.path.isdir(folder_name) or experiment in unlabeled_folders:
                continue

            patient_id = 'P' + experiment.split('-')[0][1:]
            session_id = int(experiment.split('-')[1][1:]) #'Day' + experiment.split('-')[1][1:]
            brush_type = 'Manual'
            if experiment.split('-')[6] == 'E':
                brush_type = 'Electric'
            left_right_handed = experiment.split('-')[3]
            
            meta_config_dict = {'file_name':experiment, 'patient_id':patient_id, 'session_id':session_id, 'Brush':brush_type,
                'is_left_handed':(left_right_handed == 'L'), 'Head Movement': 'No', 'Direction':1, 'Method':1}
            file_meta_config_df = pd.DataFrame(meta_config_dict, index=[0])

            files_meta_config.append(file_meta_config_df)
            
        
    files_meta_config = pd.concat(files_meta_config, ignore_index=True)
    #files_meta_config.to_csv("statssss.csv")


    return files_meta_config


def filter_files(file_name, curr_file_meta_config, eval_config):
    if file_name.startswith('.'):
        return True
    

    if eval_config['dataset_type'] == 'mine':
        patient_id, session_id = file_name.split('.')[0].split("Day")[0], int(file_name.split('.')[0].split("Day")[1])
        
        if session_id in days_poor_acc[patient_id]:
            return True

        if session_id in days_not_labeled[patient_id]:
            return True

        if  (eval_config['Head Movement'].lower() not in curr_file_meta_config['Head Movement'].values[0].lower() and eval_config['Head Movement'] != 'Both') or \
                (eval_config['Method'] != curr_file_meta_config['Method'].values[0] and eval_config['Method'] != 'Both') or \
                    (eval_config['Direction'] != curr_file_meta_config['Direction'].values[0] and eval_config['Direction'] != 'Both'):
            # instead of this we have to write the days to factor thing
            return True

    elif eval_config['dataset_type'] == 'new_paper':
        if not os.path.isdir(os.path.join(eval_config['data_path'], file_name)) or file_name in unlabeled_folders:
            return True
        
        patient_id = 'P' + file_name.split('-')[0][1:]

        if (eval_config['mode'] == 'patient_out' and  patient_id in patient_out_not_include) or \
            (eval_config['mode'] == 'session_out' and patient_id in session_out_not_include):
            return True

    if (eval_config['Brush'].lower() not in curr_file_meta_config['Brush'].values[0].lower() and eval_config['Brush'] != 'Both' and eval_config['Brush'] != 'None'):
        return True

    return False


def load_dataset_mine(data_path, file_name, eval_config, mode='train'):
    data_fields = eval_config["data_fields"].copy()
    if mode == 'train':
        data_fields_extra = data_fields + ['gyrcut_1', 'gyrcut_2', 'gyrcut_3', 'regionLabels', 'activeBrushingcut']
    elif mode == 'eval':
        data_fields_extra = data_fields + ['gyrcut_1', 'gyrcut_2', 'gyrcut_3', 'regionLabels', 'activeBrushingcut', 'segNocut']

    # here we can say based on eval config not read some types of data, e.g. brush electronic
    activeBrushExist = True
    try:
        file_content = pd.read_csv(os.path.join(data_path,file_name), usecols=data_fields_extra)
    except:
        activeBrushExist = False
        if mode == 'train':
            file_content = pd.read_csv(os.path.join(data_path,file_name), usecols=data_fields_extra[:-1])
        elif mode == 'eval':
            print(f"{file_name} doesn't have required 'activeBrushingcut' or 'segNocut' column")
            return None, None, None
    
    num_dental_regs = eval_config['num_dental_regs']
    region_map_selected = region_map
    if num_dental_regs != 16:
        region_map_selected = region_map_merge
    
    labels_sess = np.array(itemgetter(*file_content["regionLabels"].values)(region_map_selected))
    
    if activeBrushExist:
        labels_sess[np.where(file_content["activeBrushingcut"].values == 0)[0]] = np.nan

    # adding euler angles
    if eval_config['use_euler_angles'] == True:
      if 'acccut_1' in file_content.columns:
          acc=file_content[['acccut_1', 'acccut_2', 'acccut_3']].values
          gyr=file_content[['gyrcut_1', 'gyrcut_2', 'gyrcut_3']].values
          ahrsMadgw = ahrs.filters.Madgwick(acc=acc*9.8, gyr=gyr*(np.pi/180), frequency = samp_freq, beta=0.2)
          quat = ahrs.QuaternionArray(ahrsMadgw.Q) 
          euler_angs = quat.to_angles()

    features_sess = file_content[data_fields].to_numpy()
    if eval_config['use_euler_angles'] == True:
      features_sess = np.hstack([features_sess, euler_angs])

    if mode=='train':
        nan_indices = np.nonzero(np.isnan(labels_sess))[0]
        features_sess = np.delete(features_sess, nan_indices, axis=0)
        labels_sess = np.delete(labels_sess, nan_indices, axis=0)

    

    
    # file_content = shift_psi_angle(file_content, file_labels, file_name)

    return features_sess, labels_sess, file_content


    
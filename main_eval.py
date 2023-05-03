import os
import pickle
from collections import defaultdict
from operator import itemgetter, truediv
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import torch
import ahrs
from scipy.signal import find_peaks
from scipy.linalg import norm

from global_constants import region_map, region_map_merge, samp_freq
from train_validate import initialize_training
from train_validate import predict_model
from ETL import process_meta_files, filter_files, load_dataset_mine
from utils_new_paper import load_dataset_new_paper
from project_constants import samp_rate, patient_out_not_include

def CPD_extraction(file_content, dataset_type):
    new_seg_start_indices = [0]
    # diff_labels = file_content['regionLabels'].ne(file_content['regionLabels'].shift().bfill()).astype(int)
    # ground_truth_cps = list(np.where(diff_labels == 1)[0] + 1)
    
    if dataset_type == 'mine':
        gyr=file_content[['gyrcut_1', 'gyrcut_2', 'gyrcut_3']].values
        gyr_cps = find_peaks(norm(gyr, ord=2, axis=1), height=150, distance=samp_freq*2//3)[0]
        local_accAndmag_cps = list(np.where(np.diff(file_content['segNocut'].values) == 1)[0] + 1)
        mask = np.ones(len(local_accAndmag_cps))
        for index in range(len(local_accAndmag_cps)):
            local_cp = local_accAndmag_cps[index]
            if np.min(np.abs(local_cp - gyr_cps)) < 15:
                mask[index] = 0
        local_accAndmag_cps_remained = list(np.array(local_accAndmag_cps)[np.where(mask==1)[0]])
        cps = list(np.sort(local_accAndmag_cps_remained + list(gyr_cps)))
        # cps = list(local_accAndmag_cps)
        # cps = list(gyr_cps)

    elif dataset_type == 'new_paper':
        gyr = file_content[["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"]].values
        gyr_cps = find_peaks(norm(gyr, ord=2, axis=1), height=80, distance=samp_rate*2)[0]
        cps = list(gyr_cps)

    new_seg_start_indices += cps
    new_seg_start_indices += [file_content.shape[0]]
            
    return new_seg_start_indices


def eval_pipeline(training_config, eval_config, segment_config):
    
    files_meta_config = process_meta_files(eval_config['data_path'], training_config['meta_data_path'], eval_config['dataset_type'])

    classifiers_path = os.path.join(os.path.dirname(__file__), 'classifiers')
    model, _, _ = initialize_training(training_config, eval_config, segment_config)
    model.eval()

    
    with open(os.path.join(os.path.dirname(__file__), f"patient_files_dict_{eval_config['dataset_type']}"), "rb") as data_file:  # Unpickling
                patient_files = pickle.load(data_file)

    accuracies_all_files = defaultdict(defaultdict)
    labels_all_files_all_patients = []
    preds_all_files_all_patients = []
    accuracy_all_patients = []

    for patient_id in eval_config['patient_ids']:
        preds_all_files_each_patient = []
        labels_all_files_each_patient = []
            
        if eval_config['dataset_type'] == 'new_paper' and patient_id in patient_out_not_include:
            continue

        if eval_config['mode'] == 'patient_out':
            model.load_state_dict(torch.load(os.path.join(classifiers_path, f"{eval_config['dataset_type']}_{training_config['model_type']}_{eval_config['use_euler_angles']}_patient_out_{patient_id}")))
            
        for file_name in patient_files[patient_id]:
            
            curr_file_meta_config = files_meta_config[files_meta_config["file_name"]==file_name].copy()
            if filter_files(file_name, curr_file_meta_config, eval_config):
                continue
            
            brush_type = torch.tensor(int(curr_file_meta_config["Brush"].values[0] == 'Electronic')).type(torch.LongTensor)
            is_left_handed = torch.tensor(int(curr_file_meta_config["is_left_handed"].values[0])).type(torch.LongTensor)
            
            if eval_config['dataset_type'] == 'mine':
                session_id = int(file_name.split('.')[0].split("Day")[1])              
            elif eval_config['dataset_type'] == 'new_paper':
                session_id = int(file_name.split('-')[1][1:]) #'Day' + experiment.split('-')[1][1:]

            if eval_config['mode'] == 'session_out':
                model.load_state_dict(torch.load(os.path.join(classifiers_path, f"{eval_config['dataset_type']}_{training_config['model_type']}_{eval_config['use_euler_angles']}_patient_{patient_id}_session_out_{session_id}")))

            if eval_config['dataset_type'] == 'mine':
                features_sess, labels_sess, file_content = load_dataset_mine(eval_config['data_path'], file_name, eval_config, mode='eval')
            elif eval_config['dataset_type'] == 'new_paper':
                features_sess, labels_sess, file_content = load_dataset_new_paper(eval_config['data_path'], file_name, eval_config)
                file_content.loc[:, 'activeBrushingcut'] = 1

            if features_sess is None:
                continue

            ##############
            
            new_seg_start_indices = CPD_extraction(file_content, eval_config['dataset_type'])

            pred_session = []
            for i in range(len(new_seg_start_indices)-1):
                
                features_curr = features_sess[new_seg_start_indices[i]:new_seg_start_indices[i+1]]
            
                segments_all_features_curr = []
                if len(features_curr) < segment_config['segment_length']:
                    pad_range = segment_config['segment_length'] - len(features_curr)
                    segment_curr = np.pad(features_curr, ((0, pad_range), (0, 0)), mode="symmetric")
                    segments_all_features_curr.append(segment_curr)
                else:
                    for index in range(0, len(features_curr)-segment_config['segment_length']+1, segment_config['segment_stride']):
                        limit = np.arange(index, (index + segment_config['segment_length']))
                        segment_curr = features_curr[limit]

                        segments_all_features_curr.append(segment_curr)
                    segment_curr = features_curr[-segment_config['segment_length']:]
                    segments_all_features_curr.append(segment_curr)
                y_pred, p_pred = predict_model(segments_all_features_curr, model, computing_device=training_config['computing_device'], brush_type=brush_type, is_left_handed=is_left_handed)
                y_pred_features_curr = np.argmax(np.mean(p_pred, axis=0))
                pred_session.extend([y_pred_features_curr]*len(features_curr))
            ####################

            ####################
            # # segments_all_features_sess = []
            # p_pred_session = []

            # if len(features_sess) < segment_config['segment_length']:
            #     pad_range = segment_config['segment_length'] - len(features_sess)
            #     segment_curr = np.pad(features_sess, ((0, pad_range), (0, 0)), mode="symmetric")
            #     p_pred_segment_curr = np.empty((len(features_sess), eval_config['num_dental_regs']))
            #     p_pred_segment_curr[:] = np.nan
            #     p_pred_segment_curr[limit, :] = predict_model(segment_curr[np.newaxis, :, :], model, computing_device=training_config['computing_device'], brush_type=brush_type, is_left_handed=is_left_handed)[1]
            #     p_pred_session.append(p_pred_segment_curr)
            #     last_segment_len = len(features_sess)
                
            # else:
            #     for index in range(0, len(features_sess)-segment_config['segment_length']+1, segment_config['segment_stride']):
            #         limit = np.arange(index, (index + segment_config['segment_length']))
            #         segment_curr = features_sess[limit]
            #         # segments_all_features_sess.append(segment_curr)
            #         p_pred_segment_curr = np.empty((len(features_sess), eval_config['num_dental_regs']))
            #         p_pred_segment_curr[:] = np.nan
            #         p_pred_segment_curr[limit, :] = predict_model(segment_curr[np.newaxis, :, :], model, computing_device=training_config['computing_device'], brush_type=brush_type, is_left_handed=is_left_handed)[1]
            #         p_pred_session.append(p_pred_segment_curr)
                
            #     segment_curr = features_sess[-segment_config['segment_length']:]
            #     # segments_all_features_sess.append(segment_curr)
            #     last_segment_len = (len(features_sess)-segment_config['segment_length'])%segment_config['segment_stride']

            #     p_pred_segment_curr = np.empty((len(features_sess), eval_config['num_dental_regs']))
            #     p_pred_segment_curr[:] = np.nan
            #     p_pred_segment_curr[last_segment_len:, :] = predict_model(segment_curr[np.newaxis, :, :], model, computing_device=training_config['computing_device'], brush_type=brush_type, is_left_handed=is_left_handed)[1]
            #     p_pred_session.append(p_pred_segment_curr)


            # # y_pred, p_pred = predict_model(segments_all_features_sess, model, computing_device=training_config['computing_device'], brush_type=brush_type, is_left_handed=is_left_handed)
            # # repeated_y_pred = [list(itertools.repeat(element, segment_config['segment_length'])) for element in y_pred[:-1]]
            # # pred_session = list(itertools.chain.from_iterable(repeated_y_pred)) + [y_pred[-1]] * last_segment_len

            # p_pred_session = np.array(p_pred_session)
            # p_pred_session_merged = np.nanmean(p_pred_session, axis=0)
            # pred_session = np.nanargmax(p_pred_session_merged, axis=1)            
            ######################

            active_brushing_indices = np.where(file_content.loc[:, 'activeBrushingcut'].values == 1)[0]
            active_brushing_labels = labels_sess[active_brushing_indices]
            active_brushing_preds = np.array(pred_session)[active_brushing_indices]
            
            
            f1_accuracy_session = f1_score(active_brushing_labels, active_brushing_preds, average='micro')
            normal_accuracy_session = accuracy_score(active_brushing_labels, active_brushing_preds)
                        
            accuracies_all_files[patient_id][session_id] = normal_accuracy_session

            labels_all_files_each_patient.extend(active_brushing_labels)
            preds_all_files_each_patient.extend(active_brushing_preds)        
        
        labels_all_files_all_patients.extend(labels_all_files_each_patient)
        preds_all_files_all_patients.extend(preds_all_files_each_patient)
        accuracy_all_patients.append(accuracy_score(labels_all_files_each_patient, preds_all_files_each_patient))
        
        with open(os.path.join(os.path.dirname(__file__), f"accuracies_patients_{eval_config['mode']}"), "wb") as data_file:  # Pickling
            pickle.dump((accuracy_all_patients, accuracies_all_files), data_file)

    accuracy_avg_patient_out_sample_based = accuracy_score(labels_all_files_all_patients, preds_all_files_all_patients)
    # top2_accuracy_avg_patient_out_sample_based = top_k_accuracy_score(labels_all_files_all_patients, p_preds_all_files_all_patients, k=2)
    f1_avg_patient_out_sample_based = f1_score(labels_all_files_all_patients, preds_all_files_all_patients, average='micro')
    
    with open(os.path.join(os.path.dirname(__file__), f"accuracies_patients_{eval_config['mode']}"), "wb") as data_file:  # Pickling
            pickle.dump((accuracy_all_patients, accuracies_all_files, accuracy_avg_patient_out_sample_based, f1_avg_patient_out_sample_based), data_file)
    
    print(f'accuracy all patients are {accuracy_all_patients} \n\n\n')
    print(f'accuracy each file is {accuracies_all_files} \n\n\n')
    print(f'accuracy patient out sample based is {accuracy_avg_patient_out_sample_based} \n\n\n')
    print(f'f1 patient out patient based is {f1_avg_patient_out_sample_based} \n\n\n')
    

    return accuracy_avg_patient_out_sample_based, f1_avg_patient_out_sample_based
#%%
# import parallelTestModule
from statistics import mean
import sys; import os
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
except:
    pass

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score

import torch
# from adabelief_pytorch import AdaBelief
# import plotly.figure_factory as ff

from global_constants import region_names, merged_region_names, days_poor_acc
from train_test_loader import create_patient_out_loader, create_kfold_loader, create_session_out_loader
from ETL import process_meta_files, create_patients_datasets_dict, create_all_datasets, create_sessions_datasets_dict, sample_based_data_transform
from train_validate import train_model, validate_model
from main_eval import eval_pipeline
from train_validate import initialize_training, full_train_validate
from project_constants import patient_out_not_include, session_out_not_include

#%% loading the DL configs
def load_configs():
    dl_config_path = os.path.join(os.path.dirname(__file__), 'dl_constants.yaml')

    with open(dl_config_path, 'r') as file:
        configs = yaml.full_load(file)

    if configs['eval_config']['dataset_type'] == 'mine':
        configs['eval_config']['patient_ids'] = configs['eval_config']['mine_param']['patient_ids']
        configs['eval_config']['data_fields'] = configs['eval_config']['mine_param']['data_fields']
        configs['segment_config'] = configs['segment_config']['mine_param']
        
        if 'data_path' not in configs:
            # configs['eval_config']['data_path'] = os.path.join(os.path.dirname(__file__), 'data_raw/data/Phase IV') # data/Phase IV  # mixed data wth new labels and olds
            data_path = os.path.join(os.path.dirname(__file__), 'data_raw/data_with_new_labels/BrushingDataSamples') # data/Phase IV  # mixed data wth new labels and olds
            
    elif configs['eval_config']['dataset_type'] == 'new_paper':
        configs['eval_config']['patient_ids'] = configs['eval_config']['new_paper_param']['patient_ids']
        configs['eval_config']['data_fields'] = configs['eval_config']['new_paper_param']['data_fields']
        configs['segment_config'] = configs['segment_config']['new_paper_param']

        if 'data_path' not in configs:
            data_path = os.path.join(os.path.dirname(__file__), 'Paper new data/data') 
       
        
    configs['eval_config']['data_path'] = data_path

    if 'meta_data_path' not in configs:
        configs['training_config']['meta_data_path'] = os.path.join(os.path.dirname(__file__), 'data_raw/info')

    return configs

#%% plot confusion matrix
def plot_confusion_mat(labels, preds, display_labels, patient_id='random', session_id='None'):
    confusion_matrix = metrics.confusion_matrix(labels, preds, normalize = 'true')

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)

    cm_display.plot(xticks_rotation = 60)
    plt.tight_layout()

    # plt.show()

    fig_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix_plots')
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    plt.savefig(os.path.join(fig_path, f'{patient_id}_{session_id}.png'))

    return confusion_matrix

#%% functions

def train_pipeline(training_config, eval_config, segment_config):
    
    classifiers_path = os.path.join(os.path.dirname(__file__), 'classifiers')
    if not os.path.exists(classifiers_path):
        os.mkdir(classifiers_path)

    pred_labels_path = os.path.join(os.path.dirname(__file__), 'pred_and_labels')
    if not os.path.exists(pred_labels_path):
        os.mkdir(pred_labels_path)

    files_meta_config = process_meta_files(eval_config['data_path'], training_config['meta_data_path'], eval_config['dataset_type'])        
    test_predictions_all = []
    test_labels_all = []         
    test_accuracy_all = []

    if eval_config['mode'] == 'patient_out':

        patient_dataset_dict = create_patients_datasets_dict(eval_config['data_path'], files_meta_config, \
            eval_config, segment_config, seed=training_config['seed'], shuffle=True)

        
        for patient_out_id in eval_config['patient_ids']:
            print("--- Evaluating Patient ", patient_out_id,  "-----------")

            if eval_config['dataset_type'] == 'new_paper' and patient_out_id in patient_out_not_include:
                continue

            
            train_loader, valid_loader, test_loader = create_patient_out_loader(
                patient_dataset_dict,
                patient_out_id,
                batch_size=training_config['batch_size'],
                p_val=training_config['p_val'],
                seed=training_config['seed'],
                shuffle=True,
            )

            best_model, test_accuracy, test_predictions, test_labels = full_train_validate(training_config, eval_config, segment_config, train_loader, valid_loader, test_loader)
            
            if training_config['model_type'].lower() not in ['xgb', 'rf']: 
                torch.save(best_model.state_dict(), os.path.join(classifiers_path, f"{eval_config['dataset_type']}_{training_config['model_type']}_{eval_config['use_euler_angles']}_patient_out_{patient_out_id}"))

            with open(os.path.join(pred_labels_path, f"patient_out_{patient_out_id}_{eval_config['dataset_type']}_{training_config['model_type']}_test_prediction_and_labels"), "wb") as data_file:  # Pickling
                    pickle.dump((test_predictions, test_labels), data_file)

            test_predictions_all.extend(test_predictions)
            test_labels_all.extend(test_labels)   
            test_accuracy_all.append(test_accuracy)

            display_labels = region_names
            if eval_config['num_dental_regs'] != 16:
                display_labels = merged_region_names

            confusion_matrix = plot_confusion_mat(test_labels, test_predictions, display_labels, patient_id=patient_out_id)

            
            
    elif eval_config['mode'] == "k_fold":
        all_dataset = create_all_datasets(eval_config['data_path'], files_meta_config, \
            eval_config, segment_config, seed=training_config['seed'], shuffle=True)

        for i in range(eval_config['total_fold_num']):
            
            train_loader, valid_loader, test_loader = create_kfold_loader(
                all_dataset,
                i,
                eval_config['total_fold_num'],
                training_config['batch_size'],
                training_config['p_val'],
                seed=training_config['seed'],
                shuffle=True
                )

            best_model, test_accuracy, test_predictions, test_labels = full_train_validate(training_config, eval_config, segment_config, train_loader, valid_loader, test_loader)
            
            if training_config['model_type'].lower() not in ['xgb', 'rf']: 
                torch.save(best_model.state_dict(), os.path.join(classifiers_path, f"{eval_config['dataset_type']}_{training_config['model_type']}_{eval_config['use_euler_angles']}_k_fold"))

            test_predictions_all.extend(test_predictions)
            test_labels_all.extend(test_labels)   
            test_accuracy_all.append(test_accuracy)

            display_labels = region_names
            if eval_config['num_dental_regs'] != 16:
                display_labels = merged_region_names

            plot_confusion_mat(test_labels, test_predictions, display_labels)

    elif eval_config['mode'] == "session_out":
        session_dataset_dict = create_sessions_datasets_dict(eval_config['data_path'], files_meta_config, \
            eval_config, segment_config,seed=training_config['seed'], shuffle=True)

        with open(os.path.join(os.path.dirname(__file__), f"patient_files_dict_{eval_config['dataset_type']}"), "rb") as data_file:  # Unpickling
            patient_files = pickle.load(data_file)

        for patient_id in eval_config['patient_ids']:
            print("--- Working on Patient ", patient_id,  "-----------")

            if eval_config['dataset_type'] == 'new_paper' and patient_id in session_out_not_include:
                continue

            for file_name in patient_files[patient_id]:
                if eval_config['dataset_type'] == 'mine':
                    session_out_id = int(file_name.split('.')[0].split("Day")[1])
                elif eval_config['dataset_type'] == 'new_paper':
                    session_out_id = int(file_name.split('-')[1][1:]) #'Day' + experiment.split('-')[1][1:]

                print("--- Evaluating Session ", session_out_id,  "-----------")

                train_loader, valid_loader, test_loader = create_session_out_loader(
                    session_dataset_dict,
                    patient_id,
                    session_out_id,
                    batch_size=training_config['batch_size'],
                    p_val=training_config['p_val'],
                    seed=training_config['seed'],
                    shuffle=True,
                )

                best_model, test_accuracy, test_predictions, test_labels = full_train_validate(training_config, eval_config, segment_config, train_loader, valid_loader, test_loader)
            
                if training_config['model_type'].lower() not in ['xgb', 'rf']: 
                    torch.save(best_model.state_dict(), os.path.join(classifiers_path, f"{eval_config['dataset_type']}_{training_config['model_type']}_{eval_config['use_euler_angles']}_patient_{patient_id}_session_out_{session_out_id}"))

                with open(os.path.join(pred_labels_path, f"patient{patient_id}_session_out_id{session_out_id}_{eval_config['dataset_type']}_{training_config['model_type']}_test_prediction_and_labels"), "wb") as data_file:  # Pickling
                    pickle.dump((test_predictions, test_labels), data_file)
                
                test_predictions_all.extend(test_predictions)
                test_labels_all.extend(test_labels)
                test_accuracy_all.append(test_accuracy)

                display_labels = region_names
                if eval_config['num_dental_regs'] != 16:
                    display_labels = merged_region_names

                # confusion_matrix = plot_confusion_mat(test_labels, test_predictions, display_labels, patient_id=patient_id, session_id=session_out_id)
        
    print(f'\n average accuracy is {np.mean(test_accuracy_all)}')
    print(f"\n average accuracy sample based kind of is {f1_score(test_labels_all, test_predictions_all, average='micro')}")
    
#%%
if __name__ == '__main__': 
    configs = load_configs()
    print(f"model type is {configs['training_config']['model_type']} and mode is {configs['eval_config']['mode']} \n")
    train_pipeline(configs['training_config'], configs['eval_config'], configs['segment_config'])
    
    # accuracy_avg_patient_out_sample_based, f1_avg_patient_out_sample_based = \
    #     eval_pipeline(configs['training_config'], configs['eval_config'], configs['segment_config'])

    
    
    # with open(os.path.join(os.path.dirname(__file__), f"patient_files_dict_{configs['eval_config']['dataset_type']}"), "rb") as data_file:  # Unpickling
    #     patient_files = pickle.load(data_file)

    # patient_files_count = defaultdict(int)
    # for patient_id, file_names in patient_files.items(): 
    #     for file_name in file_names: 
    #         patient_id, session_id = file_name.split('.')[0].split("Day")[0], int(file_name.split('.')[0].split("Day")[1])   
    #         if session_id not in days_poor_acc[patient_id]:
    #             patient_files_count[patient_id] += 1

    # print(f'number of files of each patient is {patient_files_count}')
    
    # all_files_number = 0
    # for patient_id in patient_files_count.keys():
    #     all_files_number += patient_files_count[patient_id]

    # print(f'\n number of all files is {all_files_number}')
        



    # test_predictions_all = []
    # test_labels_all = []
    # for patient_out_id in configs['eval_config']['patient_ids']:
    #     with open(os.path.join(os.path.dirname(__file__), f"patient_out_{patient_out_id}_{configs['training_config']['model_type']}_test_prediction_and_labels"), "rb") as data_file:  # Pickling
    #         test_predictions, test_labels = pickle.load(data_file)

    #         test_predictions_all.extend(test_predictions)
    #         test_labels_all.extend(test_labels)

    # print(f"\n average accuracy sample based kind of is {f1_score(test_labels_all, test_predictions_all, average='micro')}")


    # with open(os.path.join(os.path.dirname(__file__), f"accuracies_patients_{configs['eval_config']['mode']}"), "rb") as data_file:  # Unpickling
    # # with open(os.path.join(os.path.dirname(__file__), 'accuracies_patients'), "rb") as data_file:  # Unpickling
    # #     # print(pickle.load(data_file))
    #     (accuracy_all_patients, accuracies_all_files, accuracy_avg_patient_out_sample_based, f1_avg_patient_out_sample_based) = pickle.load(data_file)
    # print(accuracy_all_patients)
    # print('\n\n')
    # print(np.mean(accuracy_all_patients))
    # print('\n\n')
    # print(accuracies_all_files)
    # print('\n\n')
    # print(accuracy_avg_patient_out_sample_based)
    # print('\n\n')
    # print(f1_avg_patient_out_sample_based)

  
    # fig_path = os.path.join(os.path.dirname(__file__), 'evaluation_plots')
    # if not os.path.exists(fig_path):
    #     os.mkdir(fig_path)
    # subject_names = [f'Subject {i}' for i in range(1,13)]
    # print(np.array(accuracy_all_patients))

    # print(np.array(accuracy_all_patients)+0.04)
    
    # plt.bar(subject_names, np.array(accuracy_all_patients)+0.04)
    # plt.xticks(rotation = '60') # Rotates X-Axis Ticks by 45-degrees
    # plt.ylabel('F1 accuracy')
    # plt.tight_layout()

    # # plt.title('Subject out')
    # plt.savefig(os.path.join(fig_path, f"accuracies_patients_{configs['eval_config']['mode']}.png"))


        

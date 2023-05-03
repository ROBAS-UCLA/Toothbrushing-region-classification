from importlib.metadata import files
from subprocess import list2cmdline
import os
from operator import itemgetter
import numpy as np

import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from data_class import RobasDataset, simpleDataset
from global_constants import region_map, region_map_merge

def create_kfold_loader(
    all_dataset,
    fold_num,
    total_fold_num,
    batch_size,
    p_val=0.1,
    seed=0,
    shuffle=True
):
    # Create the validation split from the full dataset
    length_fold = int(np.floor(len(all_dataset) / total_fold_num))
    
    test_indices = list(range(fold_num * length_fold, (fold_num+1) * length_fold))
    train_val_indices = list(set(range(0,len(all_dataset))) - set(test_indices))

    test_dataset = list(itemgetter(*test_indices)(all_dataset))
    train_val_dataset = list(itemgetter(*train_val_indices)(all_dataset))
    
    train_dataset = train_val_dataset[int(np.floor(p_val * len(train_val_dataset))):]
    val_dataset = train_val_dataset[:int(np.floor(p_val * len(train_val_dataset)))]
    
    # train_concat_dataset = Dataset_ConcatDataset(ConcatDataset(train_dataset))
    # val_concat_dataset = Dataset_ConcatDataset(ConcatDataset(val_dataset))
    # test_concat_dataset = Dataset_ConcatDataset(ConcatDataset(test_dataset))

    # train_dataset_concat = ConcatDataset(train_dataset)

    train_simple_dataset = simpleDataset(train_dataset)
    val_simple_dataset = simpleDataset(val_dataset)
    test_simple_dataset = simpleDataset(test_dataset)

    # train_loader, val_loader, test_loader = create_loader(ConcatDataset(train_dataset), ConcatDataset(val_dataset), ConcatDataset(test_dataset), batch_size)
    train_loader, val_loader, test_loader = create_loader(train_simple_dataset, val_simple_dataset, test_simple_dataset, batch_size)

    return (train_loader, val_loader, test_loader)


def create_patient_out_loader(
    patient_dataset_dict,
    patient_out_id,
    batch_size=30,
    p_val=0.1,
    seed=0,
    shuffle=True,
):
    test_dataset = patient_dataset_dict[patient_out_id]
    patient_out_dict = {k: patient_dataset_dict[k] for k in patient_dataset_dict.keys() - {patient_out_id}}
    train_val_dataset = []  
    for val in patient_out_dict.values():
        train_val_dataset.extend(val)
    
    # train_val_dataset = []
    # for patient_id in eval_config['patient_ids']:
    #     if patient_id == patient_out_id:
    #         test_dataset = [*patient_dataset_dict[patient_id]]
    #     else:
    #         train_val_dataset.extend(patient_dataset_dict[patient_id])
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(train_val_dataset)   

    train_dataset = train_val_dataset[int(np.floor(p_val * len(train_val_dataset))):]
    val_dataset = train_val_dataset[:int(np.floor(p_val * len(train_val_dataset)))]
    
    train_simple_dataset = simpleDataset(train_dataset)
    val_simple_dataset = simpleDataset(val_dataset)
    test_simple_dataset = simpleDataset(test_dataset)

    
    # train_loader, val_loader, test_loader = create_loader(ConcatDataset(train_dataset), ConcatDataset(val_dataset), ConcatDataset(test_dataset), batch_size)
    train_loader, val_loader, test_loader = create_loader(train_simple_dataset, val_simple_dataset, test_simple_dataset, batch_size)

    # print(len(train_loader), len(val_loader), len(test_loader))

    return (train_loader, val_loader, test_loader)


def create_session_out_loader(
    session_dataset_dict,
    patient_id,
    session_out_id,
    batch_size=30,
    p_val=0.1,
    seed=0,
    shuffle=True,
):
    
    test_dataset = session_dataset_dict[patient_id][session_out_id]
    session_out_dict = {k: session_dataset_dict[patient_id][k] for k in session_dataset_dict[patient_id].keys() - {session_out_id}}
    train_val_dataset = []
    for val in session_out_dict.values():
        train_val_dataset.extend(val)    


    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(train_val_dataset)   


    train_dataset = train_val_dataset[int(np.floor(p_val * len(train_val_dataset))):]
    val_dataset = train_val_dataset[:int(np.floor(p_val * len(train_val_dataset)))]
    

    train_simple_dataset = simpleDataset(train_dataset)
    val_simple_dataset = simpleDataset(val_dataset)
    test_simple_dataset = simpleDataset(test_dataset)

    
    # train_loader, val_loader, test_loader = create_loader(train_dataset, val_dataset, test_dataset, batch_size)
    train_loader, val_loader, test_loader = create_loader(train_simple_dataset, val_simple_dataset, test_simple_dataset, batch_size)

    # print(len(train_loader), len(val_loader), len(test_loader))

    return (train_loader, val_loader, test_loader)



def create_loader(train_dataset, val_dataset, test_dataset, batch_size = 64):

    num_workers = 0
    pin_memory = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


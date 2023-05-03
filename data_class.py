import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class RobasDataset(Dataset):
    
    def __init__(
        self,
        features,
        label,
        patient_id,
        session_id,
        file_meta_config, 
        segment_length = 50,
        segment_stride = 5,
        transform = transforms.ToTensor()
    ):
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.features = features
        self.transform = transform

        self.label = label
        self.patient_id = patient_id
        self.session_id = session_id
        self.brush_type = int(file_meta_config["Brush"].values[0] == 'Electronic')
        self.with_head_movement = int(file_meta_config["Head Movement"].values[0] == 'Yes')
        self.direction = int(file_meta_config["Direction"].values[0] == 1) #direction 2 will become 0
        self.method = int(file_meta_config["Method"].values[0] == 1) # 1 is sequential and 2 is freestyle which will become 0
        
        self.is_left_handed = int(file_meta_config["is_left_handed"].values[0])
        # self.__add_extra_features()
        self.__preprocessing()
        #self.filter_invalid()

    def __add_extra_features(self):
        extra_features = np.zeros((len(self.features), 3))
        extra_features[:,0] = self.brush_type
        extra_features[:,1] = self.with_head_movement
        extra_features[:,2] = self.is_left_handed
        self.features = np.hstack([self.features, extra_features])

    def __preprocessing(self):
        self.annotations = []
        if len(self.features) < self.segment_length:
            pad_range = self.segment_length - len(self.features)
            segment = np.pad(
                self.features,
                ((0, pad_range), (0, 0)),
                mode="symmetric"
            )
            new_annot = {'data':segment, 'label':self.label, \
                    'patient_id':self.patient_id, 'session_id':self.session_id, 'brush_type':self.brush_type, 'is_left_handed':self.is_left_handed}
            self.annotations.append(new_annot)

        else:
            for index in range(0, len(self.features)-self.segment_length+1, self.segment_stride):
                limit = np.arange(index, (index + self.segment_length))
                segment = self.features[limit]
                new_annot = {'data':segment, 'label':self.label, \
                    'patient_id':self.patient_id, 'session_id':self.session_id, 'brush_type':self.brush_type, 'is_left_handed':self.is_left_handed}
                self.annotations.append(new_annot)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]['data'], self.annotations[idx]['label'], self.annotations[idx]['brush_type'], self.annotations[idx]['is_left_handed']
            # self.annotations[idx]['patient_id'], self.annotations[idx]['session_id']


class simpleDataset(Dataset):
    
    def __init__(
        self,
        dataset,
        transform = transforms.ToTensor()
    ):
        # self.__add_extra_features()
        self.dataset = dataset
        #self.filter_invalid()

   
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.dataset[idx][3]




# class Dataset_ConcatDataset(Dataset):
    
#     def __init__(
#         self,
#         concat_dataset,
#         transform = transforms.ToTensor()
#     ):
#         # self.__add_extra_features()
#         self.__preprocessing(concat_dataset)
#         #self.filter_invalid()

    
#     def __preprocessing(self, concat_dataset):

#         self.all_features = []
#         self.all_labels = []
#         self.all_brush_type = []
#         self.all_is_left_handed = []
#         for index in range(len(concat_dataset)):
#             features_curr = concat_dataset[index][0]
#             self.all_features.append(features_curr)
#             self.all_labels.extend([concat_dataset[index][1]]*features_curr.shape[0])
#             self.all_brush_type.extend([concat_dataset[index][2]]*features_curr.shape[0])
#             self.all_is_left_handed.extend([concat_dataset[index][3]]*features_curr.shape[0])


#         self.all_features = np.vstack(self.all_features)
#         self.all_labels = np.array(self.all_labels)
#         self.all_brush_type = np.array(self.all_brush_type)
#         self.all_is_left_handed = np.array(self.all_is_left_handed)
    
#         nan_indices = np.nonzero(np.isnan(self.all_labels))[0]
#         self.all_labels = np.delete(self.all_labels, nan_indices, axis=0)
#         self.all_features = np.delete(self.all_features, nan_indices, axis=0)
#         self.all_brush_type = np.delete(self.all_brush_type, nan_indices, axis=0)
#         self.all_is_left_handed = np.delete(self.all_is_left_handed, nan_indices, axis=0)
        
#     def __len__(self):
#         return len(self.all_labels)

#     def __getitem__(self, idx):
#         return self.all_features[idx], self.all_labels[idx], self.all_brush_type[idx], self.all_is_left_handed[idx]

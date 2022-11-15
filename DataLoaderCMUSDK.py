import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from Utils import whether_type_str
from Config import Data_path_SDK

# MOSI Structure
mosi_l_features = ["text", "glove", "last_hidden_state", "masked_last_hidden_state", "pooler_output", "summed_last_four_states"]
mosi_a_features = ["covarep", "opensmile_eb10", "opensmile_is09"]
mosi_v_features = ["facet41", "facet42", "openface"]
# [[l_features, a_features, v_features], _label, _label_2, _label_7, segment]
    
# MOSEI Structure
mosei_l_features = ["text", "glove", "last_hidden_state", "masked_last_hidden_state", "pooler_output", "summed_last_four_states"]
mosei_a_features = ["covarep"]
mosei_v_features = ["facet42"]
# [[l_features, a_features, v_features], _label, _label_2, _label_7, segment]

# POM Structure
pom_l_features = ["text", "glove", "last_hidden_state", "masked_last_hidden_state", "pooler_output", "summed_last_four_states"]
pom_a_features = ["covarep"]
pom_v_features = ["facet42"]
# [[l_features, a_features, v_features], _label, _label_7, segment]

DATA_PATH = Data_path_SDK

def multi_collate_mosei_mosi(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.Tensor([sample[3] for sample in batch]).reshape(-1,).float()
    labels_2 = torch.Tensor([sample[4] for sample in batch]).reshape(-1,).long()
    if whether_type_str(batch[0][0][0]):
        sentences = [sample[0].tolist() for sample in batch]
    else:
        sentences = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch], padding_value=0).transpose(0, 1)
    acoustic = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch], padding_value=0).transpose(0, 1)
    visual = pad_sequence([torch.FloatTensor(sample[2]) for sample in batch], padding_value=0).transpose(0, 1)
        
    lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])
    return sentences, acoustic, visual, labels, labels_2, lengths

def get_mosi_dataset(mode='train', text='glove', audio='covarep', video='facet42', normalize=[True, True, True]):
    with open(os.path.join(DATA_PATH, 'mosi_'+mode+'.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    assert text in mosi_l_features 
    assert audio in mosi_a_features 
    assert video in mosi_v_features
    l_features = [np.nan_to_num(data_[0][0][mosi_l_features.index(text)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    a_features = [np.nan_to_num(data_[0][1][mosi_a_features.index(audio)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    v_features = [np.nan_to_num(data_[0][2][mosi_v_features.index(video)], nan=0.0, posinf=0, neginf=0) for data_ in data]

    if normalize[0]:
        max_l, min_l = max([np.max(f) for f in l_features]), min([np.min(f) for f in l_features])
        l_features = [2*(f-min_l)/(max_l-min_l)-1 for f in l_features]
    if normalize[1]:
        max_a, min_a = max([np.max(f) for f in a_features]), min([np.min(f) for f in a_features])
        a_features = [2*(f-min_a)/(max_a-min_a)-1 for f in a_features]
    if normalize[2]:
        max_v, min_v = max([np.max(f) for f in v_features]), min([np.min(f) for f in v_features])
        v_features = [2*(f-min_v)/(max_v-min_v)-1 for f in v_features]
        
    labels = [data_[1] for data_ in data]
    labels_2 = [data_[2] for data_ in data]
    
    return l_features, a_features, v_features, labels, labels_2

def get_mosei_dataset(mode='train', text='glove', audio='covarep', video='facet42', normalize=[True, True, True]):
    with open(os.path.join(DATA_PATH, 'mosei_'+mode+'.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    assert text in mosei_l_features  
    assert audio in mosei_a_features  
    assert video in mosei_v_features
    l_features = [np.nan_to_num(data_[0][0][mosei_l_features.index(text)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    a_features = [np.nan_to_num(data_[0][1][mosei_a_features.index(audio)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    v_features = [np.nan_to_num(data_[0][2][mosei_v_features.index(video)], nan=0.0, posinf=0, neginf=0) for data_ in data]

    if normalize[0]:
        max_l, min_l = max([np.max(f) for f in l_features]), min([np.min(f) for f in l_features])
        l_features = [2*(f-min_l)/(max_l-min_l)-1 for f in l_features]
    if normalize[1]:
        max_a, min_a = max([np.max(f) for f in a_features]), min([np.min(f) for f in a_features])
        a_features = [2*(f-min_a)/(max_a-min_a)-1 for f in a_features]
    if normalize[2]:
        max_v, min_v = max([np.max(f) for f in v_features]), min([np.min(f) for f in v_features])
        v_features = [2*(f-min_v)/(max_v-min_v)-1 for f in v_features]
        
    labels = [data_[1] for data_ in data]
    labels_2 = [data_[2] for data_ in data]
    labels_7 = [data_[3] for data_ in data]
    
    return l_features, a_features, v_features, labels, labels_2, labels_7

class CMUSDKDataset(Dataset):
    def __init__(self, mode, dataset='mosi', text='glove', audio='covarep', video='facet42', normalize=[True, True, True]):
        assert mode in ['test', 'train', 'valid']
        assert dataset in ['mosei', 'mosi', 'pom']

        self.dataset = dataset
        if dataset == 'mosi':
            self.l_features, self.a_features, self.v_features, self.labels, self.labels_2 = get_mosi_dataset(mode=mode, text=text, audio=audio, video=video, normalize=normalize)
        elif dataset == 'mosei':
            self.l_features, self.a_features, self.v_features, self.labels, self.labels_2, self.labels_7 = get_mosei_dataset(mode=mode, text=text, audio=audio, video=video, normalize=normalize)
        else:
            raise NotImplementedError

    def __getitem__(self, index):        
        if self.dataset == 'mosi':
            return self.l_features[index], self.a_features[index], self.v_features[index], self.labels[index], self.labels_2[index]
        elif self.dataset == 'mosei':
            return self.l_features[index], self.a_features[index], self.v_features[index], self.labels[index], self.labels_2[index], self.labels_7[index]
        else:
            raise NotImplementedError
            
    def __len__(self):
        return len(self.labels)

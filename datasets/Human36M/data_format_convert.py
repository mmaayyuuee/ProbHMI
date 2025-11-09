import numpy as np
import copy
import os

'''
一个一次性脚本, 当制作data_3d_h36m_expmap.npz后将不再使用
'''

def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
    filename: string. Path to the csv file
    Returns
    returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    # returnArray = np.reshape(returnArray, (returnArray.shape[0], -1, 3))
    return returnArray


def load_data(path_to_dataset, subjects, actions, one_hot):
    """
    Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Args
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
    Returns
    trainData: dictionary with k:v
        k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
    completeData: nxd matrix with all the data. Used to normlization stats
    """
    nactions = len( actions )

    trainData = {}
    completeData = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]

            for subact in [1, 2]:  # subactions
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)
                action_sequence = readCSVasFloat(filename)

                n, d = action_sequence.shape
                even_list = range(0, n, 1)
                
                if one_hot:
                    # Add a one-hot encoding at the end of the representation
                    the_sequence = np.zeros( (len(even_list), d + nactions), dtype=float )
                    the_sequence[ :, 0:d ] = action_sequence[even_list, :]
                    the_sequence[ :, d+action_idx ] = 1
                    trainData[(subj, action, subact, 'even')] = the_sequence
                else:
                    trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]

                if len(completeData) == 0:
                    completeData = copy.deepcopy(action_sequence)
                else:
                    completeData = np.append(completeData, action_sequence, axis=0)
    return trainData, completeData


# save_dir = os.path.join(os.getcwd(), 'data', 'data_3d_h36m_expmap.npz')
# data = np.load(save_dir, allow_pickle=True)
# expmap = data['expmap_3d'].item()
# bone_length = data['bone_length'].item()
# print()


# 数据转存
actions = ["walking", "eating", "smoking", "discussion",  "directions", "greeting", "phoning", "posing", "purchases", 
           "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]
train_subject_ids = [1, 5, 6, 7, 8, 9, 11]
data_dir = os.path.join(os.getcwd(), 'data', 'h36m_dataset')

train_set, complete_train = load_data(data_dir, train_subject_ids, actions, False)

expmap_data = {}
for keys, value in train_set.items():
    subject, action, subact, _ = keys
    subject = 'S' + str(subject)
    action = action + str(subact)
    if subject not in expmap_data:
        expmap_data[subject] = {}
    expmap_data[subject][action] = value


data_pos_dir = os.path.join(os.getcwd(), 'data', 'data_3d_h36m.npz')
pos_data = np.load(data_pos_dir, allow_pickle=True)
pos_data = pos_data['positions_3d'].item()


import sys
sys.path.append(os.getcwd())
from utils.skeleton import Skeleton

skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                            16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                    joints_left=[6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23],
                    joints_right=[1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31])

import torch
def get_bones_length(data):
    parents = torch.tensor(skeleton.parents(), requires_grad=False)
    parents[0] = 0
    bones_length = {}
    for subject, dict_s in data.items():
        seqs = []
        for _, seq in dict_s.items():
            seqs.append(seq)
        seqs = np.concatenate(seqs, 0)
        data = torch.tensor(seqs, requires_grad=False)
        bone_st = data[:, 1:, :]            
        bone_ed = torch.index_select(data, 1, parents)[:, 1:, :]
        diff = torch.subtract(bone_ed, bone_st)
        dist = torch.sum(torch.pow(diff, 2), dim=-1)
        bone_length = torch.sqrt(dist)
        bone_length = torch.mean(bone_length, dim=0).numpy()  
        bones_length[subject] = np.zeros(shape=(32,))
        bones_length[subject][1:] = bone_length
    return bones_length
bones_length = get_bones_length(pos_data)

mask  =  np.array([0,  0, 0,
                  -1,  0, 0,
                   0, -1, 0,
                   0, -1, 0,
                   0,  0, 1,
                   0,  0, 1,
                   1,  0, 0,
                   0, -1, 0,
                   0, -1, 0,
                   0,  0, 1,
                   0,  0, 1,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  0, 0,
                   0,  0, 1,
                   0,  1, 0,
                   0,  0, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  1, 0,
                   0,  0, 0,
                   0,  0, 1,
                   0,  1, 0,
                   0,  0, 0,])
mask = mask.reshape(-1,3)

bones_length_mat = {}
for key, bone_length in bones_length.items():
    input = np.broadcast_to(np.reshape(bone_length, (32,1)), shape=(32, 3))
    bone_length_mat = np.multiply(mask, input)
    bones_length_mat[key] = bone_length_mat


save_dir = os.path.join(os.getcwd(), 'data', 'data_3d_h36m_expmap.npz')
np.savez(save_dir, expmap_3d=expmap_data, bone_length=bones_length_mat)
print()
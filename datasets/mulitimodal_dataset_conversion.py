import numpy as np
import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
from scipy.spatial.distance import pdist, squareform
    
from datasets.HumanEva.dataset_humaneva import DataSetHumanEVAexpMap
from datasets.Human36M.dataset_h36m_17n import DataSetH36MexpMap18nodes



human36m_dataset_root = os.path.join(os.getcwd(), 'data', 'data_3d_h36m_expmap.npz')
human36m_dataset = DataSetH36MexpMap18nodes(human36m_dataset_root, 'test', t_his=25, t_pred=100, actions='all')

humanEva_dataset_root = os.path.join(os.getcwd(), 'data', 'data_3d_humaneva_expmap.npz')
humanEva_dataset = DataSetHumanEVAexpMap(humanEva_dataset_root, 'test', t_his=15, t_pred=60, actions='all')


def get_multimodal_gt(dataset, data_gen, t_his):
    all_data = []
    for data, label in data_gen:
        data = np.swapaxes(data, -2, -1)
        data = dataset.convert_to_position_mode(data)
        data = np.reshape(data, newshape=(1, data.shape[0], -1))
        all_data.append(data)
        
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < 0.5)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
    return traj_gt_arr


human36m_generator = human36m_dataset.iter_generator(step=25, batch_size=1)
get_multimodal_gt(human36m_dataset, human36m_generator, 25)

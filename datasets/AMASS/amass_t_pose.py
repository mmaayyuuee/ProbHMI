import torch
import numpy as np

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c


models_dir = os.path.join(os.getcwd(), "data/auxiliary")
save_dir = os.path.join(os.getcwd(), "data/auxiliary/AMASS/amass_t_poses.npz")

# ## test
# t_poses = np.load(save_dir)
# t_pose_male = t_poses['male']
# t_pose_female = t_poses['female']

genders = ("male", "female")
t_poses = {}
for gender in genders:
    bm_fname = os.path.join(models_dir, 'smplh/{}/model.npz'.format(gender))

    smpl_dict = np.load(bm_fname, encoding='latin1')
    v_template = smpl_dict['v_template']
    J_regressor = smpl_dict['J_regressor']
    t_pose = np.matmul(J_regressor, v_template)
    
    t_pose_aug = np.zeros(shape=(24, 3))
    t_pose_aug[0:22] = t_pose[0:22]
    # t_pose_aug[0] = 0.0
    t_poses[gender] = t_pose_aug
    # print(t_pose[0:22])

np.savez(save_dir, **t_poses)
print()
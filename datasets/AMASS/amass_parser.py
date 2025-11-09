import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
import argparse
import tarfile
from io import BytesIO

import numpy as np
import torch
import zarr
from tqdm import tqdm


os.environ['PYOPENGL_PLATFORM'] = 'egl' # https://github.com/mmatl/pyrender/issues/13
TARGET_OPEN_GL_MAJOR = 3
TARGET_OPEN_GL_MINOR = 3


def process_data(path, out, target_fps):
    z_poses = zarr.open(os.path.join(out, 'poses.zarr'), mode='w', shape=(0, 24, 3), chunks=(1000, 24, 3), dtype=np.float32)
    z_trans = zarr.open(os.path.join(out, 'trans.zarr'), mode='w', shape=(0, 3), chunks=(1000, 3), dtype=np.float32)
    z_index = zarr.open(os.path.join(out, 'poses_index.zarr'), mode='w', shape=(0, 2), chunks=(1000, 2), dtype=int)
    z_gender = zarr.open(os.path.join(out, 'poses_gender.zarr'), mode='w', shape=(0, 1), chunks=(1000, 1), dtype=int)
    i = 0
    tar = tarfile.open(path, 'r')
    for member in tqdm(tar):
        file_name = os.path.basename(member.name)
        if file_name.endswith('.npz') and not file_name.startswith('.'):
            try:
                with tar.extractfile(member) as f:
                    array_file = BytesIO()
                    array_file.write(f.read())
                    array_file.seek(0)
                    bdata = np.load(array_file)

                    if 'mocap_framerate' not in bdata and 'mocap_frame_rate' not in bdata:
                        print(f"WARNING: we skip '{member.name}' because it is corrupted (no framerate)")
                        continue
                    else:
                        frame_rate = bdata['mocap_framerate'] if 'mocap_framerate' in bdata else bdata['mocap_frame_rate']
                    gender = str(bdata["gender"])
                    if gender == "b'female'":
                        gender = "female" # this is a common problem in SSM dataset

                    if target_fps == -1:
                        fps = frame_rate
                    else:
                        fps = target_fps

                    frame_multiplier = int(np.round(frame_rate / fps))
                    
                    body_joints_ori = bdata['poses'][:, :66][::frame_multiplier]
                    body_joints_ori = np.reshape(body_joints_ori, newshape=(-1, 22, 3))
                    
                    body_joints = np.zeros(shape=(body_joints_ori.shape[0], 24, 3))
                    body_joints[:, :22, :] = body_joints_ori
                    
                    body_trans = bdata['trans'][::frame_multiplier]
                    
                    body_gender = np.array([[0]]) if gender == 'female' else np.array([[1]])
                    
                    z_poses.append(body_joints, axis=0)
                    z_trans.append(body_trans, axis=0)
                    z_index.append(np.array([[i, i + body_joints.shape[0]]]), axis=0)
                    z_gender.append(body_gender, axis=0)
                    i = i + body_joints.shape[0]
            except Exception as e:
                print(e, ". Filename:", file_name)
                


parser = argparse.ArgumentParser(description='AMASS Process Raw Data')
parser.add_argument('--fps',
                    type=int,
                    default=60,
                    help='FPS')
parser.add_argument('--datasets',
                    type=str,
                    nargs="+",
                    help='The names of the datasets to process',
                    default=None)
parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')
args = parser.parse_args()


# python -m data_loader.parsers.amass --gpu
# it will pre-process the AMASS dataset
if __name__ == '__main__':
    fps = args.fps
    datasets = args.datasets

    in_path = os.path.join(os.getcwd(), "data/AMASS_Original")
    out_path = os.path.join(os.getcwd(), "data/AMASS")

    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using device:", comp_device)
    
    if datasets is None:
        # we process all datasets in 'in_path' folder
        datasets = sorted([p.split(".")[0] for p in os.listdir(in_path)])

    # list all datasets
    print("Datasets to process:")
    print(datasets)
    for i, dataset in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] Processing {dataset}...")
        process_data(os.path.join(in_path, dataset + '.tar.bz2'), os.path.join(out_path, dataset), fps)

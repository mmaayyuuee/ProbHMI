import torch
import numpy as np
import math


def onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes).to(y.device)
    if len(y.size()) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1), 1)
    elif len(y.size()) == 2:
        y_onehot = y_onehot.scatter_(1, y, 1)
    else:
        raise ValueError("[onehot]: y should be in shape [B], or [B, C]")
    return y_onehot


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor
    

def cat_feature(input_a, input_b, type="graph"):
    if type == "graph":
        return torch.add(input_a, input_b)
    elif type == "tensor":
        return torch.cat((input_a, input_b), dim=1)    


def timesteps(tensor):
    return int(tensor.size(2))


# from np.array to torch.tensor and from (B, N, C) to (B, C, N)
def vis2nn_dformat(x):
    assert x.ndim == 2 or x.ndim == 3 or x.ndim == 4 or x.ndim == 5
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = torch.transpose(x, -2, -1)
    return x


# from torch.tensor to np.array and from (B, C, N) to (B, N, C)
def nn2vis_dformat(x):
    assert x.ndim == 2 or x.ndim == 3 or x.ndim == 4 or x.ndim == 5
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    x = np.swapaxes(x, -2, -1)
    return x


def euler2rotmat_torch(euler):
    """
    将 ZYX 欧拉角 (Yaw-Pitch-Roll) 转换为旋转矩阵
    :param euler: (..., 3) 欧拉角 (yaw, pitch, roll) in radians
    :return: (..., 3, 3) 旋转矩阵
    """
    yaw = euler[..., 0]
    pitch = euler[..., 1]
    roll = euler[..., 2]
    
    # 计算三角函数值
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    
    # 构造旋转矩阵 (ZYX顺序: 先绕Z轴(yaw)，然后绕Y轴(pitch)，最后绕X轴(roll))
    R = torch.zeros(euler.shape[:-1] + (3, 3), device=euler.device, dtype=euler.dtype)
    
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    code from https://github.com/enriccorona/human-motion-prediction-pytorch/blob/master/src/data_utils.py
    Args
    R: a 3x3 rotation matrix
    Returns
    eul: a 3x1 Euler angle representation of R
    """
    if R.ndim == 2:
        eul = np.zeros(shape=(3,), dtype=np.float32)
    elif R.ndim == 3:
        eul = np.zeros(shape=(R.shape[0], 3), dtype=np.float32)
    elif R.ndim == 4:
        eul = np.zeros(shape=(R.shape[0], R.shape[1], 3), dtype=np.float32)
    elif R.ndim == 5:
        eul = np.zeros(shape=(R.shape[0], R.shape[1], R.shape[2], 3), dtype=np.float32)
    elif R.ndim == 6:
        eul = np.zeros(shape=(R.shape[0], R.shape[1], R.shape[2], R.shape[3], 3), dtype=np.float32)
    
    mask_base = R[..., 0, 2]
    mask1 = np.where((mask_base == 1.0) | (mask_base == -1.0), 1.0, 0.0)
    mask2 = np.where( mask_base == 1.0, 1.0, 0.0)
    
    dlta = np.arctan2(R[..., 0, 1], R[..., 0, 2])
    
    eul1 = np.zeros_like(eul)
    eul1[..., 0] = dlta
    eul1[..., 1] = np.pi/2
    eul1[..., 2] = 0.0

    eul2 = np.zeros_like(eul)
    eul2[..., 0] = dlta
    eul2[..., 1] = -np.pi/2
    eul2[..., 2] = 0.0
    
    eul3 = np.zeros_like(eul)
    E2 = -np.arcsin( R[..., 0, 2] )
    E1 = np.arctan2( R[..., 1, 2]/np.cos(E2), R[..., 2, 2]/np.cos(E2) )
    E3 = np.arctan2( R[..., 0, 1]/np.cos(E2), R[..., 0, 0]/np.cos(E2) )    
    eul3[..., 0] = E1
    eul3[..., 1] = E2
    eul3[..., 2] = E3

    mask1, mask2 = mask1[..., np.newaxis], mask2[..., np.newaxis]
    eul = eul + eul3 * (1 - mask1) + eul2 * mask2 + eul1 * (mask1 - mask2)
    return eul


def rotmat2euler_torch(R):
    """
    将旋转矩阵转换为 ZYX 欧拉角 (Yaw-Pitch-Roll)
    :param R: (..., 3, 3) 旋转矩阵
    :return: (..., 3) 欧拉角 (yaw, pitch, roll) in radians
    """
    pitch = torch.asin(-R[..., 2, 0])

    # 处理万向节锁 (Gimbal Lock)
    gimbal_mask = torch.abs(pitch - math.pi/2) < 1e-6
    neg_gimbal_mask = torch.abs(pitch + math.pi/2) < 1e-6

    # 一般情况
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])

    # 万向节锁时，固定 roll=0，计算 yaw
    yaw[gimbal_mask] = torch.atan2(-R[..., 0, 1][gimbal_mask], R[..., 1, 1][gimbal_mask])
    roll[gimbal_mask] = 0.0

    yaw[neg_gimbal_mask] = torch.atan2(R[..., 0, 1][neg_gimbal_mask], R[..., 1, 1][neg_gimbal_mask])
    roll[neg_gimbal_mask] = 0.0
    return torch.stack((yaw, pitch, roll), dim=-1)


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    code from https://github.com/enriccorona/human-motion-prediction-pytorch/blob/master/src/data_utils.py
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """

    if q.ndim == 1:
        q = np.expand_dims(q, 0)
    if q.ndim == 2:
        q = np.expand_dims(q, 0)    

    if True in (np.where(np.abs(np.linalg.norm(q, axis=-1)-1)>1e-3, True, False)):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    _sinhalftheta = np.linalg.norm(q[..., 1:], axis=-1)
    sinhalftheta = np.expand_dims(_sinhalftheta, -1)
    coshalftheta = q[..., :1]

    r0 = np.divide(q[..., 1:], (sinhalftheta + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2*np.pi, 2*np.pi)

    theta_mask = np.where(theta > np.pi, False, True)
    r0_mask = np.broadcast_to(theta, shape=r0.shape)
    
    theta = np.where(theta_mask is False, 2*np.pi-theta, theta)
    r0 = np.where(r0_mask is False, -r0, r0)
    r = r0 * theta
    r = np.squeeze(r)
    return r


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    code from https://github.com/enriccorona/human-motion-prediction-pytorch/blob/master/src/data_utils.py
    Args
    R: 3x3 rotation matrix
    Returns
    q: 1x4 quaternion
    """
    if R.ndim == 2:
        R = np.expand_dims(R, 0)
    if R.ndim == 3:
        R = np.expand_dims(R, 0)
    
    rotdiff = R - np.transpose(R, (0, 1, 3, 2))
    
    r = np.zeros(shape=(R.shape[0], R.shape[1], 3))
    r[..., 0] = rotdiff[..., 2, 1]
    r[..., 1] = rotdiff[..., 0, 2]
    r[..., 2] = rotdiff[..., 1, 0]
    
    r_n = np.expand_dims(np.linalg.norm(r, axis=-1), -1)
    r0 = np.divide(r, r_n + np.finfo(np.float32).eps)
    
    sintheta = r_n / 2
    costheta = (np.trace(R, axis1=-1, axis2=-2) - 1) / 2
    costheta = np.expand_dims(costheta, -1)
    theta = np.arctan2(sintheta, costheta)
    
    q = np.zeros(shape=(R.shape[0], R.shape[1], 4))
    q[..., :1] = np.cos(theta/2)
    q[..., 1:] = r0*np.sin(theta/2)
    q = np.squeeze(q)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def rotmat2expmap_torch(R):
    """
    将旋转矩阵转换为指数映射表示
    参数:
        R: 旋转矩阵, shape (..., 3, 3)
    返回:
        expmap: 指数映射表示, shape (..., 3)
    """
    # 确保输入是torch张量
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32)
    
    # 计算旋转角度
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # 处理小角度情况
    small_angle = theta < 1e-6
    large_angle = ~small_angle
    
    # 初始化输出
    expmap = torch.zeros_like(R[..., 0, :3])
    
    # 大角度情况
    if large_angle.any():
        # 计算旋转轴
        R_large = R[large_angle]
        theta_large = theta[large_angle].unsqueeze(-1)
        # 避免数值不稳定
        w1 = R_large[..., 2, 1] - R_large[..., 1, 2]
        w2 = R_large[..., 0, 2] - R_large[..., 2, 0]
        w3 = R_large[..., 1, 0] - R_large[..., 0, 1]
        w = torch.stack([w1, w2, w3], dim=-1)
        # 归一化并乘以角度
        w_norm = torch.norm(w, dim=-1, keepdim=True)
        w_normalized = w / w_norm
        expmap_large = w_normalized * theta_large
        expmap[large_angle] = expmap_large
    
    # 小角度情况（使用泰勒展开近似）
    if small_angle.any():
        R_small = R[small_angle]
        # 直接使用反对称矩阵的提取方式
        w1 = R_small[..., 2, 1] - R_small[..., 1, 2]
        w2 = R_small[..., 0, 2] - R_small[..., 2, 0]
        w3 = R_small[..., 1, 0] - R_small[..., 0, 1]
        expmap_small = torch.stack([w1, w2, w3], dim=-1) / 2
        expmap[small_angle] = expmap_small
    return expmap


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    code from https://github.com/enriccorona/human-motion-prediction-pytorch/blob/master/src/data_utils.py
    Args
    r: 1x3 exponential map
    Returns
    R: 3x3 rotation matrix
    """
    antisymmetry = np.array(
            [0,  0,  0,   0,  0, -1,    0, 1, 0,
             0,  0,  1,   0,  0,  0,   -1, 0, 0,
             0, -1,  0,   1,  0,  0,    0, 0, 0]
        ).reshape(9, 3).T
    
    if r.ndim == 1:
        r = np.expand_dims(r, 0)
    if r.ndim == 2:
        r = np.expand_dims(r, 0)
    
    theta = np.linalg.norm(r, axis=-1)
    _theta = np.broadcast_to(np.expand_dims(theta, axis=-1), r.shape)
    
    r0 = np.divide(r, _theta + np.finfo(np.float32).eps)
    r0x = np.dot(r0, antisymmetry)
    
    if r0x.ndim == 3:
        r0x = np.reshape(r0x, (r0x.shape[0], r0x.shape[1], -1, 3))
    elif r0x.ndim == 4:
        r0x = np.reshape(r0x, (r0x.shape[0], r0x.shape[1], r0x.shape[2], -1, 3))
    elif r0x.ndim == 5:
        r0x = np.reshape(r0x, (r0x.shape[0], r0x.shape[1], r0x.shape[2], r0x.shape[3], -1, 3))
    
    _theta = np.broadcast_to(np.expand_dims(_theta, axis=-1), r0x.shape)
    
    sin_theta, cos_theta = np.sin(_theta), np.cos(_theta)
    R = np.eye(3,3) + sin_theta*r0x + (1-cos_theta)*(np.matmul(r0x, r0x))
    R = np.squeeze(R)
    return R


def expmap2rotmat_torch(r):
    """
    code from https://github.com/MotionMLP/MotionMixer/blob/91327c3c3a455d398bd097fa300385bafa80a835/utils/data_utils.py
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3, device=r.device).repeat(n, 1, 1).float() + \
        torch.mul(torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + \
        torch.mul((1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def expmap2rotmat_torch_V2(r):
    antisymmetry = torch.tensor(
        [0,  0,  0,   0,  0, -1,    0, 1, 0,
         0,  0,  1,   0,  0,  0,   -1, 0, 0,
         0, -1,  0,   1,  0,  0,    0, 0, 0], dtype=torch.float32, device=r.device).reshape(9, 3).T

    theta = torch.norm(r, dim=-1)
    _theta = theta.unsqueeze(-1).expand_as(r)

    r0 = r / (_theta + torch.finfo(torch.float32).eps)
    r0x = torch.matmul(r0, antisymmetry)
    r0x_new_shape = (*r0x.shape[:-1], r0x.shape[-1] // 3, 3)
    r0x = r0x.reshape(r0x_new_shape)

    _theta = _theta.unsqueeze(-1).expand_as(r0x)

    sin_theta, cos_theta = torch.sin(_theta), torch.cos(_theta)
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + sin_theta * r0x + (1 - cos_theta) * (torch.matmul(r0x, r0x))
    return R


def forward_kinematics(expmaps, parents, offsets):
    origin_shape = expmaps.shape
    if len(origin_shape) == 2:
        expmaps = np.expand_dims(expmaps, [0,1])
    elif len(origin_shape) == 3:
        expmaps = np.expand_dims(expmaps, 0)
    expmaps = np.reshape(expmaps, (-1, expmaps.shape[-2], expmaps.shape[-1]))
    
    njoints = expmaps.shape[-2]
    xyzStruct = [dict() for _ in range(njoints)]
    
    for i in range(njoints):
        r = expmaps[:, i:i+1, :]
        rotation = expmap2rotmat(r)
        
        if parents[i] == -1:
            xyzStruct[i]['rotation'] = rotation
            xyzStruct[i]['xyz'] = np.broadcast_to(offsets[i], (expmaps.shape[0], offsets[i].shape[0]))
        else:
            xyzStruct[i]['xyz'] = np.matmul(offsets[i], xyzStruct[parents[i]]['rotation']) + xyzStruct[parents[i]]['xyz']
            xyzStruct[i]['rotation'] = np.matmul(rotation, xyzStruct[parents[i]]['rotation'])
            
    xyz = [np.expand_dims(xyzStruct[i]['xyz'], 1) for i in range(njoints)]
    xyz = np.concatenate(xyz, 1)
    xyz = xyz[..., [0,2,1]]
    xyz = np.reshape(xyz, origin_shape)
    return xyz


def fkl_torch(angles, parent, offset):
    if angles.ndim == 3:
        n = angles.shape[0]
    elif angles.ndim == 4:
        n = angles.shape[0] * angles.shape[1]
    elif angles.ndim == 5:
        n = angles.shape[0] * angles.shape[1] * angles.shape[2]
    j_n = offset.shape[0]
    
    p3d = torch.from_numpy(offset).float().unsqueeze(0).repeat(n, 1, 1).to(angles.device)
    # angles = angles[:, 3:].contiguous().view(-1, 3)
    _angles = torch.reshape(angles, (-1, 3))
    R = expmap2rotmat_torch(_angles).view(n, j_n, 3, 3).float()
    for i in np.arange(1, j_n):
        if parent[i] >= 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    p3d = torch.reshape(p3d, angles.shape)
    p3d = p3d[..., [0,2,1]]
    return p3d


def vel2pose(ff, vel, batch_order=False):
    pose = torch.zeros_like(vel, device=vel.device)
    if batch_order:
        length = vel.shape[1]
        for i in range(length):
            if i == 0:
                pose[:, i, :, :] = ff[:, -1, :, :] + vel[:, i, :, :]
            else:
                pose[:, i, :, :] = pose[:, i-1, :, :] + vel[:, i, :, :]
    else:
        length = vel.shape[0]
        for i in range(length):
            if i == 0:
                pose[i] = ff[-1] + vel[i]
            else:
                pose[i] = pose[i-1] + vel[i]
    return pose



def rotmat_to_sixd(rotmat):
    """
    将旋转矩阵转换为6D表示
    :param rotmat: [..., 3, 3] 旋转矩阵
    :return: [..., 6] 6D表示
    """
    # 提取前两列
    rot_6d = rotmat[..., :, :2]
    # 重塑为6D向量
    rot_6d = rot_6d.reshape(*rotmat.shape[:-2], 6)
    return rot_6d


def rotmat_to_sixd_torch(rotmat):
    """
    PyTorch版本的旋转矩阵到6D转换
    """
    rot_6d = rotmat[..., :, :2]
    rot_6d = rot_6d.view(*rotmat.shape[:-2], 6)
    return rot_6d


def sixd_to_rotmat_torch(rot_6d):
    """
    将6D表示转换为旋转矩阵
    :param rot_6d: [..., 6] 6D表示
    :return: [..., 3, 3] 旋转矩阵
    """
    # 重塑为3x2矩阵
    rot_6d = rot_6d.reshape(*rot_6d.shape[:-1], 3, 2)
    # 计算第三个轴（叉积）
    x = rot_6d[..., :, 0]  # 第一列
    y = rot_6d[..., :, 1]  # 第二列
    # 确保正交性
    x = x / torch.norm(x, dim=-1, keepdim=True)
    z = torch.cross(x, y, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    # 构建旋转矩阵
    rotmat = torch.stack([x, y, z], dim=-1)
    return rotmat


def sixd_to_rotmat(rot_6d):
    """
    NumPy版本的6D到旋转矩阵转换
    """
    rot_6d = rot_6d.reshape(*rot_6d.shape[:-1], 3, 2)
    
    x = rot_6d[..., :, 0]
    y = rot_6d[..., :, 1]
    
    # 归一化第一个轴
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    # 计算第三个轴（叉积）
    z = np.cross(x, y, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    # 重新计算第二个轴以确保正交性
    y = np.cross(z, x, axis=-1)
    rotmat = np.stack([x, y, z], axis=-1)
    return rotmat
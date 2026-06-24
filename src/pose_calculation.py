import numpy as np

def anchor_to_hmd_pose(anchor_T, hmd_T):
    """
    Compute HMD pose relative to anchor marker.
    anchor_T: 4x4 transform matrix of anchor marker (ID 100+) in camera space
    hmd_T:    4x4 transform matrix of HMD marker (ID 0-99) in camera space
    returns:  4x4 transform matrix of HMD pose in anchor space
    """
    return np.linalg.inv(anchor_T) @ hmd_T
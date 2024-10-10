import numpy as np
import h5py
import os
import time
import open3d as o3d

def loadh5(filedir, color_format='rgb'):
    """Load coords & feats from h5 file.

    Arguments: file direction

    Returns: coords & feats.
    """
    pc = h5py.File(filedir, 'r')['data'][:]
    print(type(pc),pc.shape,pc[0],pc[:,1])
    coords = pc[:, 0:3].astype('int32')

    if color_format == 'rgb':
        feats = pc[:, 3:6] / 255.
    elif color_format == 'yuv':
        R, G, B = pc[:, 3:4], pc[:, 4:5], pc[:, 5:6]
        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
        Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
        Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
        feats = np.concatenate((Y, Cb, Cr), -1) / 256.
    elif color_format == 'geometry':
        feats = np.expand_dims(np.ones(coords.shape[0]), 1)
    elif color_format == 'None':
        return coords

    feats = feats.astype('float32')
    print(coords.shape[0])
    return coords, feats


def loadply(filedir, color_format='geometry'):
    """Load coords & feats from ply file.

    Arguments: file direction.

    Returns: coords & feats.

      Data descriptors defined here:
 |
 |  colors
 |      ``float64`` array of shape ``(num_points, 3)``, range ``[0, 1]`` , use ``numpy.asarray()`` to access data: RGB colorsof points.
 |
 |  covariances
 |      ``float64`` array of shape ``(num_points, 3, 3)``, use ``numpy.asarray()`` to access data: Points covariances.
 |
 |  normals
 |      ``float64`` array of shape ``(num_points, 3)``, use ``numpy.asarray()`` to access data: Points normals.
 |
 |  points
 |      ``float64`` array of shape ``(num_points, 3)``, use ``numpy.asarray()`` to access data: Points coordinates.

    """
    pcd = o3d.io.read_point_cloud(filedir)
    print(pcd)
    coords = np.asarray(pcd.points)
    feats = np.asarray(pcd.colors)

    pc_data = np.concatenate([coords, feats], axis=-1)
    print(coords.shape,feats.shape,pc_data.shape)
    return coords, feats, pc_data

coords, feats,pc = loadply('S26C03R03_rec_0536.ply')
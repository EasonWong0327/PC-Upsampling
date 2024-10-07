
import time, os, sys, glob, argparse

dataset_8i = '/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldier/Ply/'
dataset_8i2 = '/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/longdress/Ply/'

eighti_filedirs = glob.glob(dataset_8i + '*.ply')
eighti_filedirs2 = glob.glob(dataset_8i2 + '*.ply')
print(len(eighti_filedirs),len(eighti_filedirs2))
import numpy as np
import os
import shutil

from evaluate_orb import evaluate_orb
from evaluate_eloftr import evaluate_eloftr
from utils import get_image_list

def process_image_list(trajectory_name, image_list):
    #camera intrinsics
    #see: https://github.com/castacks/tartanair_tools/blob/master/data_type.md
    fx = 320.0
    fy = 320.0
    cx = 320.0
    cy = 240.0
    K0 = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    K0 = np.array(K0, dtype="float32")
    K1 = K0

    evaluate_orb(trajectory_name, image_list, K0, K1)
    evaluate_eloftr(trajectory_name, image_list, K0, K1)

if __name__ == '__main__':

    #load images
    image_list_seasidetown = get_image_list("/home/renato/workspace/Datasets/TartanAir/seasidetown_sample_P003/P003/image_left", 0, 334)
    image_list_carwelding = get_image_list("/home/renato/workspace/Datasets/TartanAir/carwelding_sample_P007/P007/image_left", 0, 356)

    if os.path.exists("results"):
        shutil.rmtree("results")

    process_image_list("seasidetown", image_list_seasidetown)
    process_image_list("carwelding", image_list_carwelding)
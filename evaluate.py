import cv2
import numpy as np
import os
import time
import csv
import shutil

from copy import deepcopy
from eloftr.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from eloftr.src.utils.plotting import make_matching_figure

import torch
import matplotlib.cm as cm

def estimate_pose(kpts0, kpts1, K0, K1, thresh=0.5, conf=0.99999):
    if len(kpts0) < 5:
        return None
    
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 5
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n >= best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def get_image_list(left_base_path, min_idx, max_idx):
    image_list = []
    for idx in range(min_idx, max_idx):
        full_path = f"{left_base_path}/{idx:06}_left.png"
        image_list.append(full_path)

    return image_list

def evaluate_orb(trajectory_name, image_list, K0, K1):
    results_data = [ 
        ["idx0", "idx1", "exec_time_ms", "kpts0_count", "kpts1_count", "inliers_count"]
    ]

    #create ORB detector
    method_name = "orb"
    max_features = 5000
    orb = cv2.ORB_create(max_features)

    #process first image in a set against the others in the set
    print(f"***** Evaluating {method_name} in trajectory {trajectory_name} *****")
    set_len = 10
    for idx0 in range(0, len(image_list) - set_len, set_len):
        print(f"Processing set {idx0:03}-{idx0+set_len:03}")
        for idx1 in range(idx0+1, idx0+1+set_len):        
            img0_pth = image_list[idx0]
            img1_pth = image_list[idx1]

            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

            start_time = time.perf_counter()

            #Detected keypoints and descriptors on each image
            kpts0, desc0 = orb.detectAndCompute(img0_raw,None)
            kpts1, desc1 = orb.detectAndCompute(img1_raw,None)

            if len(kpts0) < 5 or len(kpts1) < 5: #discard poor detections
                continue

            #Match keypoints on both images and cross check 
            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            orb_matches = bf_matcher.match(desc0, desc1)
            
            #sort matches by distance (shorter is more confident)
            orb_matches = sorted(orb_matches, key=lambda val: val.distance)

            end_time = time.perf_counter()
            exec_time_ms = (end_time - start_time) * 1000 

            #creates a list of matches keypoints
            matched_kpts0 = np.array([kpts0[match.queryIdx].pt for match in orb_matches]) 
            matched_kpts1 = np.array([kpts1[match.trainIdx].pt for match in orb_matches]) 

            #estimate pose between two images with matching kp, up to scale
            ret = estimate_pose(matched_kpts0, matched_kpts1, K0, K1)
            if ret is None:
                continue
            else:
                R, t, inliers = ret
                inliers_count = np.count_nonzero(inliers)

                #filter matches to those supporting estimated pose
                filtered_orb_matches = []
                for i in range(len(inliers)):
                    is_match = inliers[i]
                    if is_match:
                        filtered_orb_matches.append(orb_matches[i])

                #Output results
                folder_path = f"results/{trajectory_name}/{method_name}/{idx0:03}-{idx0+set_len:03}"
                os.makedirs(folder_path, exist_ok=True)
                os.makedirs(f'{folder_path}/unfiltered', exist_ok=True)        
                os.makedirs(f'{folder_path}/filtered', exist_ok=True)        

                #Draw matching figure
                unfiltered_img_match = cv2.drawMatches(img0_raw, kpts0, img1_raw, kpts1, orb_matches,None)
                cv2.imwrite(f'{folder_path}/unfiltered/{idx0:03}-{idx1:03}.png', unfiltered_img_match)    

                filtered_img_match = cv2.drawMatches(img0_raw, kpts0, img1_raw, kpts1, filtered_orb_matches,None)
                cv2.imwrite(f'{folder_path}/filtered/{idx0:03}-{idx1:03}.png', filtered_img_match)

                #Append results
                results_data.append([idx0, idx1, exec_time_ms, len(kpts0), len(kpts1), inliers_count])     

    with open(f"results/{trajectory_name}/{method_name}/{method_name}_results_log.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write each row to the CSV file
        for row in results_data:
            writer.writerow(row)

def evaluate_eloftr(trajectory_name, image_list, K0, K1):
    results_data = [ 
        ["idx0", "idx1", "exec_time_ms", "kpts0_count", "kpts1_count", "inliers_count"]
    ]
        
    #create EfficientLoFTR detector
    method_name = "eloftr"
    # You can choose model type in ['full', 'opt']
    model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

    # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
    precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

    # You can also change the default values like thr. and npe (based on input image size)

    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
        
    if precision == 'mp':
        _default_cfg['mp'] = True
    elif precision == 'fp16':
        _default_cfg['half'] = True    
    
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("/home/renato/workspace/EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
    matcher = reparameter(matcher) # no reparameterization will lead to low performance

    if precision == 'fp16':
        matcher = matcher.half()
    matcher = matcher.eval().cuda()

    #process first image in a set against the others in the set
    print(f"***** Evaluating {method_name} in trajectory {trajectory_name} *****")
    set_len = 10
    for idx0 in range(0, len(image_list) - set_len, set_len):
        print(f"Processing set {idx0:03}-{idx0+set_len:03}")
        for idx1 in range(idx0+1, idx0+1+set_len):        
            img0_pth = image_list[idx0]
            img1_pth = image_list[idx1]

            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

            if precision == 'fp16':
                img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
            else:
                img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
                img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

            batch = {'image0': img0, 'image1': img1}

            # Inference with EfficientLoFTR and get prediction
            with torch.no_grad():
                start_time = time.perf_counter()
                if precision == 'mp':
                    with torch.autocast(enabled=True, device_type='cuda'):
                        matcher(batch)
                else:
                    matcher(batch)
                end_time = time.perf_counter()
                exec_time_ms = (end_time - start_time) * 1000 
                
                kpts0 = batch['mkpts0_f'].cpu().numpy()
                kpts1 = batch['mkpts1_f'].cpu().numpy()
                conf = batch['mconf'].cpu().numpy()

            #Output results
            folder_path = f"results/{trajectory_name}/{method_name}/{idx0:03}-{idx0+set_len:03}"
            os.makedirs(folder_path, exist_ok=True)
            os.makedirs(f'{folder_path}/unfiltered', exist_ok=True)        
            os.makedirs(f'{folder_path}/filtered', exist_ok=True)        

            # Draw matching figure
            if model_type == 'opt':
                conf = (conf - min(20.0, conf.min())) / (max(30.0, conf.max()) - min(20.0, conf.min()))

            color = cm.jet(conf)
            text = [
                'LoFTR',
                'Matches: {}'.format(len(kpts0)),
            ]

            unfiltered_img_path = f"{folder_path}/unfiltered/{idx0:03}-{idx1:03}.png"
            make_matching_figure(img0_raw, img1_raw, kpts0, kpts1, color, text=text, path=unfiltered_img_path)

            #Append results
            results_data.append([idx0, idx1, exec_time_ms, len(kpts0), len(kpts1), 0])     

    with open(f"results/{trajectory_name}/{method_name}/{method_name}_results_log.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write each row to the CSV file
        for row in results_data:
            writer.writerow(row)

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

    #evaluate_orb(trajectory_name, image_list, K0, K1)
    evaluate_eloftr(trajectory_name, image_list, K0, K1)

if __name__ == '__main__':

    #load images
    image_list_seasidetown = get_image_list("/home/renato/workspace/Datasets/TartanAir/seasidetown_sample_P003/P003/image_left", 0, 334)
    image_list_carwelding = get_image_list("/home/renato/workspace/Datasets/TartanAir/carwelding_sample_P007/P007/image_left", 0, 356)

    if os.path.exists("results"):
        shutil.rmtree("results")

    process_image_list("seasidetown", image_list_seasidetown)
    process_image_list("carwelding", image_list_carwelding)
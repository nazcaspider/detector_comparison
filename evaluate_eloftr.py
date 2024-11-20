import cv2
import numpy as np
import os
import time
import csv
import torch
import matplotlib.cm as cm

from copy import deepcopy
from eloftr.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from eloftr.src.utils.plotting import make_matching_figure

from utils import estimate_pose

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
                
                matched_kpts0 = batch['mkpts0_f'].cpu().numpy()
                matched_kpts1 = batch['mkpts1_f'].cpu().numpy()
                conf = batch['mconf'].cpu().numpy()

            #estimate pose between two images with matching kp, up to scale and filter
            ret = estimate_pose(matched_kpts0, matched_kpts1, K0, K1)
            if ret is None:
                continue
            else:
                R, t, inliers = ret
                inliers_count = np.count_nonzero(inliers)


            #filter matches to those supporting estimated pose
            filtered_matched_kpts0 = []
            filtered_matched_kpts1 = []
            filtered_conf = []
            for i in range(len(inliers)):
                is_match = inliers[i]
                if is_match:
                    filtered_matched_kpts0.append(matched_kpts0[i])
                    filtered_matched_kpts1.append(matched_kpts1[i])  
                    filtered_conf.append(conf[i])  

            filtered_matched_kpts0 = np.array(filtered_matched_kpts0)
            filtered_matched_kpts1 = np.array(filtered_matched_kpts1)
            filtered_conf = np.array(filtered_conf)

            #Output results
            folder_path = f"results/{trajectory_name}/{method_name}/{idx0:03}-{idx0+set_len:03}"
            os.makedirs(folder_path, exist_ok=True)
            os.makedirs(f'{folder_path}/unfiltered', exist_ok=True)        
            os.makedirs(f'{folder_path}/filtered', exist_ok=True)        

            # Draw matching figure
            if model_type == 'opt':
                conf = (conf - min(20.0, conf.min())) / (max(30.0, conf.max()) - min(20.0, conf.min()))

            color = cm.jet(conf)
            filtered_color = cm.jet(filtered_conf)

            unfiltered_text = [
                'LoFTR',
                'Unfiltered Matches: {}'.format(len(matched_kpts0)),
            ]

            filtered_text = [
                'LoFTR',
                'Filtered Matches: {}'.format(len(filtered_matched_kpts0)),
            ]

            unfiltered_img_path = f"{folder_path}/unfiltered/{idx0:03}-{idx1:03}.png"
            make_matching_figure(img0_raw, img1_raw, matched_kpts0, matched_kpts1, color, text=unfiltered_text, path=unfiltered_img_path)

            filtered_img_path = f"{folder_path}/filtered/{idx0:03}-{idx1:03}.png"
            make_matching_figure(img0_raw, img1_raw, filtered_matched_kpts0, filtered_matched_kpts1, filtered_color, text=filtered_text, path=filtered_img_path)

            #Append results
            results_data.append([idx0, idx1, exec_time_ms, len(matched_kpts0), len(matched_kpts1), len(filtered_matched_kpts0)])     

    with open(f"results/{trajectory_name}/{method_name}/{method_name}_results_log.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write each row to the CSV file
        for row in results_data:
            writer.writerow(row)
import argparse
import json
import os
import random

from matplotlib.colors import LinearSegmentedColormap
from flow_checker import visualize_optical_flow
import torch
import cv2
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from models.mem_cvae import HFVAD
from scipy.ndimage import median_filter
from datasets.dataset import Chunked_sample_dataset, get_dataset
from utils.eval_utils import load_and_resize_gt_images, robust_scale, save_aupr_fpr_curve, save_evaluation_curves, save_flow_images, save_frame_plots
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
from sklearn.metrics import f1_score
from torchvision.transforms import ToPILImage

METADATA = {
 "carla_local": {
        "testing_video_num": 13,
        "testing_frames_cnt": [93, 100, 93, 93, 93, 100, 100, 100, 100, 100, 200, 200, 200]
    },
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}






def evaluate(config, ckpt_path, testing_chunked_samples_file, training_stats_path, suffix):
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    device = config["device"]
    num_workers = config["num_workers"]

    testset_num_frames = np.sum(METADATA[dataset_name]["testing_frames_cnt"])

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  ).to(device).eval()

    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    print("load pre-trained success!")

    if training_stats_path is not None:
        training_scores_stats = torch.load(training_stats_path)

        of_mean, of_std = np.mean(training_scores_stats["of_training_stats"]), \
                          np.std(training_scores_stats["of_training_stats"])
        frame_mean, frame_std = np.mean(training_scores_stats["frame_training_stats"]), \
                                np.std(training_scores_stats["frame_training_stats"])

    score_func = nn.MSELoss(reduction="none")

    dataset_test = Chunked_sample_dataset(testing_chunked_samples_file)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=num_workers, shuffle=False)
    dataset = get_dataset(dataset_name=dataset_name,
                        dir=os.path.join('', dataset_name),
                        context_frame_num=1, mode="test")

    # bbox anomaly scores for each frame
    frame_bbox_scores = [{} for i in range(testset_num_frames.item())]
    all_bboxes_test = np.load('carla_final_test_bboxes.npy', allow_pickle=True)

    original_size = (768, 512) 
    target_size = (192, 128) 

    pixel_gt_dict = load_and_resize_gt_images(dataset.all_gt_addr, target_size)
    pixel_scores = {}
    global_index = 0
    for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):
        sample_frames_test, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
        batch_frame_numbers = pred_frame_test.numpy()
        
        sample_frames_test = sample_frames_test.to(device)
        sample_ofs_test = sample_ofs_test.to(device)

        out_test = model(sample_frames_test, sample_ofs_test, mode="test")
        grouped_flow_preds = []
        grouped_flow_targets = []
        grouped_preds = []
        grouped_targets = []

        # Show frame predictions of model
        for i, frame_number in enumerate(batch_frame_numbers):
                global_i = global_index + i
                frame_number = frame_number.item()

                grouped_preds.append(out_test["frame_pred"][i])
                grouped_targets.append(out_test["frame_target"][i])
                # comment if you dont want to save the model predictions and flows
                #save_images_with_diff(grouped_preds, grouped_targets, global_i, frame_number, config)
                grouped_flow_preds.append(out_test["of_recon"][i])
                grouped_flow_targets.append(out_test["of_target"][i])

                #save_flow_images(grouped_flow_preds, grouped_flow_targets, global_i,frame_number, config)
                grouped_flow_preds = []
                grouped_flow_targets = []
                grouped_preds = []
                grouped_targets = []

        global_index += len(batch_frame_numbers)


        mse_of_test = score_func(out_test["of_recon"], out_test["of_target"]).cpu().data.numpy()
        mse_frame_test = score_func(out_test["frame_pred"], out_test["frame_target"]).cpu().data.numpy()
       
        normalized_mse_of_test = robust_scale(mse_of_test)
        normalized_mse_frame_test = robust_scale(mse_frame_test)

        aggregated_mse_of_test = normalized_mse_of_test.mean(axis=1, keepdims=True)  
        aggregated_mse_frame_test = normalized_mse_frame_test.mean(axis=1, keepdims=True)  

        pixelwise_anomaly_score = config["w_r_p"] * aggregated_mse_of_test + config["w_p_p"] * aggregated_mse_frame_test

        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

        for frame_idx, frame_number in enumerate(pred_frame_test.numpy()):
            frame_number_scalar = frame_number.item()
            for y in range(target_size[1]):  
                for x in range(target_size[0]):
                    pixel_key = (frame_number_scalar, y, x)
                    pixel_scores[pixel_key] = 0
            
        for frame_idx, frame_number in enumerate(pred_frame_test.numpy()):
                frame_number_scalar = frame_number.item()
                for bbox in all_bboxes_test[frame_number_scalar]:
                    x_min, y_min, x_max, y_max = bbox  

                    scaled_x_min = int(x_min * scale_x)
                    scaled_y_min = int(y_min * scale_y)
                    scaled_x_max = int(x_max * scale_x)
                    scaled_y_max = int(y_max * scale_y)
                    
                    current_frame_scores = pixelwise_anomaly_score[frame_idx, :, :, :].max(axis=0)
                    scaled_anomaly_scores = cv2.resize(current_frame_scores, ((scaled_x_max - scaled_x_min), (scaled_y_max - scaled_y_min)), interpolation=cv2.INTER_LINEAR)
                    
                    for x in range(scaled_x_min, scaled_x_max):
                        for y in range(scaled_y_min, scaled_y_max):
                            pixel_key = (frame_number_scalar, y, x)
                            score_x = x - scaled_x_min
                            score_y = y - scaled_y_min
                            if score_x < scaled_anomaly_scores.shape[1] and score_y < scaled_anomaly_scores.shape[0]:
                                pixel_scores[pixel_key] = scaled_anomaly_scores[score_y, score_x]

                    
        # frame-wise
        loss_of_test = score_func(out_test["of_recon"], out_test["of_target"]).cpu().data.numpy()
        loss_frame_test = score_func(out_test["frame_pred"], out_test["frame_target"]).cpu().data.numpy()

        of_scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1)
        frame_scores = np.sum(np.sum(np.sum(loss_frame_test, axis=3), axis=2), axis=1)

        if training_stats_path is not None:
            # mean-std normalization
            of_scores = (of_scores - of_mean) / of_std
            frame_scores = (frame_scores - frame_mean) / frame_std

        scores = config["w_r"] * of_scores + config["w_p"] * frame_scores
        for i in range(len(scores)):
            frame_bbox_scores[pred_frame_test[i][-1].item()][i] = scores[i]
    

    # pixel wise anomaly score
    scores_list = []
    labels_list = []
    # Missing scores gets replaced by 0
    keys_in_pixel_gt_dict_not_in_pixel_scores = set(pixel_gt_dict.keys()) - set(pixel_scores.keys())
    for key in keys_in_pixel_gt_dict_not_in_pixel_scores:
        if key not in pixel_scores:
            pixel_scores[key] = 0

    assert set(pixel_scores.keys()) == set(pixel_gt_dict.keys()), "Error with the keys"

    for key in pixel_scores.keys():
        scores_list.append(pixel_scores[key])
        labels_list.append(pixel_gt_dict[key])

   # turn on to save pixel wise plots
    #for i in range(testset_num_frames):
     #  save_frame_plots(i, pixel_scores, pixel_gt_dict, target_size, config)

    scores_array = np.array(scores_list)
    labels_array = np.array(labels_list)

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'AUPR_FPR' , suffix)
    save_aupr_fpr_curve(scores_array, labels_array, curves_save_path)

    # frame-level anomaly score
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
                frame_scores[i] = config["w_r"] * (0 - of_mean) / of_std + config["w_p"] * (0 - frame_mean) / frame_std
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores,
                os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix))
    
    # ================== Calculate AUC ==============================
    with open("carla_data/ground_truth_demo/gt_label.json", "r") as file:
        gt = json.load(file)
    gt_concat = np.concatenate(list(gt.values()), axis=0)


    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        gt_each_video = gt_concat[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        scores_each_video = frame_scores[
                            start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]

        start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str,
                        default="/disk/vanishing_data/hb344/finetune/ckpt/carla_local_ML_MemAE_SC_with_follower_data/best.pth",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="/fzi/ids/hb344/no_backup/Desktop/hf2vad/cfgs/finetune_cfg.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))
    testing_chunked_samples_file = os.path.join("./data", config["dataset_name"],
                                                "testing/chunked_samples/chunked_samples_00.pkl")
    
    testing_chunked_samples_file = 'carla_data/STC/testing/chunked_samples/chunked_samples_00.pkl'

    from train import cal_training_stats

    os.makedirs(os.path.join("./eval", config["exp_name"]), exist_ok=True)
    #training_chunked_samples_dir = os.path.join("./data", config["dataset_name"], "training/chunked_samples")
    #training_stat_path = os.path.join("./eval", config["exp_name"], "training_stats.npy")
    #training_stat_path = '/disk/vanishing_data/hb344/finetune/ckpt/carla_local_ML_MemAE_SC_with_follower_data/training_stats.npy-70'
    #training_stat_path = '/media/hb344/53E9-5477/CVAE_best/training_stats.npy-40'
    training_chunked_samples_dir = '/disk/vanishing_data/hb344/STC'
    training_stat_path = 'training_stats_finetune.npy'
    #cal_training_stats(config, args.model_save_path, training_chunked_samples_dir, training_stat_path)

    with torch.no_grad():
        auc = evaluate(config, args.model_save_path,
                       testing_chunked_samples_file,
                       training_stat_path, suffix="best")

        print("AUC", auc)

import json
import os
import torch
from torch.utils.data import DataLoader
import cv2
import torch.nn as nn
import numpy as np
import yaml
import joblib
import pickle
import gc
from tqdm import tqdm

from datasets.dataset import Chunked_sample_dataset, get_dataset
from models.ml_memAE_sc import ML_MemAE_SC
from utils.eval_utils import load_and_resize_gt_images, save_aupr_fpr_curve, save_aupr_fpr_curve_1, save_evaluation_curves

METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
"carla_local": {
        "testing_video_num": 13,
        "testing_frames_cnt": [93, 100, 93, 93, 93, 100, 100, 100, 100, 100, 200, 200, 200]
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


def evaluate(config, ckpt_path, testing_chunked_samples_file, suffix):
    dataset_name = config["dataset_name"]
    device = config["device"]
    num_workers = config["num_workers"]
    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    testset_num_frames = np.sum(METADATA[dataset_name]["testing_frames_cnt"])

    os.makedirs(eval_dir, exist_ok=True)

    model = ML_MemAE_SC(num_in_ch=config["model_paras"]["motion_channels"],
                        seq_len=config["model_paras"]["num_flows"],
                        features_root=config["model_paras"]["feature_root"],
                        num_slots=config["model_paras"]["num_slots"],
                        shrink_thres=config["model_paras"]["shrink_thres"],
                        mem_usage=config["model_paras"]["mem_usage"],
                        skip_ops=config["model_paras"]["skip_ops"]).to(device).eval()

    # load weights
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")

    dataset_test = Chunked_sample_dataset(testing_chunked_samples_file, last_flow=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=num_workers, shuffle=False)
    dataset = get_dataset(dataset_name=dataset_name,
                        dir=os.path.join('', dataset_name),
                        context_frame_num=1, mode="test")
    target_size = (192, 128) 
    original_size = (768, 512) 

    # bbox anomaly scores for each frame
    frame_bbox_scores = [{} for i in range(testset_num_frames.item())]
    all_bboxes_test = np.load('carla_final_test_bboxes.npy', allow_pickle=True)
    pixel_gt_dict = load_and_resize_gt_images(dataset.all_gt_addr, target_size)
    pixel_scores = {}

    for ii, test_data in tqdm(enumerate(dataloader_test), desc="Eval: ", total=len(dataloader_test)):
        _, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
        sample_ofs_test = sample_ofs_test.cuda()

        out_test = model(sample_ofs_test)

        pixelwise_errors = (out_test["recon"] - sample_ofs_test) ** 2

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
                
                current_frame_errors = pixelwise_errors[frame_idx, :, :, :].max(axis=0)
            
                current_frame_errors_np = current_frame_errors.cpu().detach().numpy()
                
                scaled_anomaly_errors = cv2.resize(current_frame_errors_np, ((scaled_x_max - scaled_x_min), (scaled_y_max - scaled_y_min)), interpolation=cv2.INTER_LINEAR)
                
                for x in range(scaled_x_min, scaled_x_max):
                    for y in range(scaled_y_min, scaled_y_max):
                        pixel_key = (frame_number_scalar, y, x)
                        score_x = x - scaled_x_min
                        score_y = y - scaled_y_min
                        pixel_scores[pixel_key] = scaled_anomaly_errors[score_y, score_x]

        loss_of_test = score_func(out_test["recon"], sample_ofs_test).cpu().data.numpy()
        scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1)

        for i in range(len(scores)):
            frame_bbox_scores[pred_frame_test[i][-1].item()][i] = scores[i]

    del dataset_test
    gc.collect()

    joblib.dump(frame_bbox_scores, os.path.join(config["eval_root"], config["exp_name"],
                                                "frame_bbox_scores_%s.json" % suffix))
    
    keys_in_pixel_gt_dict_not_in_pixel_scores = set(pixel_gt_dict.keys()) - set(pixel_scores.keys())
    for key in keys_in_pixel_gt_dict_not_in_pixel_scores:
        if key not in pixel_scores:
            pixel_scores[key] = 0

    assert set(pixel_scores.keys()) == set(pixel_gt_dict.keys()), "Error with the keys"

    # pixel wise anomaly score
    scores_list = []
    labels_list = []
    for key in pixel_scores.keys():
        scores_list.append(pixel_scores[key])
        labels_list.append(pixel_gt_dict[key])
    
    scores_array = np.array(scores_list)
    labels_array = np.array(labels_list)

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'AUPR_FPR' , suffix)
    save_aupr_fpr_curve(scores_array, labels_array, curves_save_path)
    # frame_bbox_scores = joblib.load(os.path.join(config["eval_root"], config["exp_name"],
    #                                              "frame_bbox_scores_%s.json" % suffix))

    # frame-level anomaly score (i.e., the maximum anomaly scores of all the objects in it)
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
            frame_scores[i] = 0  # assign ZERO when no object exists
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores,
                os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix))

    # frame_scores = joblib.load(
    #     os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix)
    # )

    # ================== Calculate AUC ==============================
    # load gt labels
    #gt = pickle.load(
     #   open(os.path.join(config["dataset_base_dir"], "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
    with open("carla_data/ground_truth_demo/gt_label.json", "r") as file:
        gt = json.load(file)
    gt_concat = np.concatenate(list(gt.values()), axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        cur_video_len = METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

        gt_each_video = gt_concat[start_idx:start_idx + cur_video_len][4:]
        scores_each_video = frame_scores[start_idx:start_idx + cur_video_len][4:]

        start_idx += cur_video_len

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

    return auc


if __name__ == '__main__':
    model_save_path = "/disk/vanishing_data/hb344/ML_memeAE/ckpt/carla_local_ML_MemAE_SC_with_follower_data/best.pth"
    cfg_file = "cfgs/ml_memAE_sc_cfg.yaml"

    config = yaml.safe_load(open(cfg_file))
    dataset_base_dir = config["dataset_base_dir"]
    dataset_name = config["dataset_name"]

    #testing_chunked_samples_file = os.path.joitestset_num_framestestset_num_framestestset_num_framesn("./data", config["dataset_name"],
     #                                           "testing/chunked_samples/chunked_samples_00.pkl")
    
    testing_chunked_samples_file = '/disk/users/hb344/no_backup/Desktop/hf2vad/carla_data/STC/testing/chunked_samples/chunked_samples_00.pkl'


    with torch.no_grad():
        auc = evaluate(config, model_save_path, testing_chunked_samples_file, suffix="best")
        print(auc)

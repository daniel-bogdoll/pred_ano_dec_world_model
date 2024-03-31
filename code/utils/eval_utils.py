import json
import numpy as np
import os
import matplotlib
import torch
from flow_vis import flow_to_color 
import pickle
from torchvision.transforms import ToPILImage



from typing import Tuple
import cv2

# Set the backend to 'pgf'
#matplotlib.use('pgf')
#matplotlib.rcParams.update({
 #   "axes.unicode_minus": False,  # Use ASCII hyphen-minus rather than Unicode minus
  #  "pgf.texsystem": "pdflatex",
   #"font.family": "serif",
   #"text.usetex": True,
   #"pgf.rcfonts": False,
   # "pgf.preamble": [
    #    r"\usepackage[utf8]{inputenc}",  # Ensure LaTeX understands UTF-8 input
    #    r"\usepackage[T1]{fontenc}",     # Use a font that provides the Unicode characters
     #   r"\usepackage{lmodern}",         # Use the Latin Modern font
   #]#
#})

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import scipy.signal as signal
import seaborn as sns


def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(psnr_dir, "auc.png"))
 #   plt.savefig(os.path.join(psnr_dir, "auc.pgf"))
    plt.close()


def save_aupr_fpr_curve(scores, labels, curves_save_path):
    if not os.path.exists(curves_save_path):
        os.makedirs(curves_save_path, exist_ok=True)
    
    np.save(os.path.join(curves_save_path, 'raw_scores.npy'), scores)
    np.save(os.path.join(curves_save_path, 'raw_labels.npy'), labels)
    # Saving KDE data
    scores_label_1 = scores[labels == 1]
    scores_label_0 = scores[labels == 0]
    kde_data = {
        'scores_label_1': scores_label_1.tolist(),
        'scores_label_0': scores_label_0.tolist()
    }

    with open(os.path.join(curves_save_path, "kde_data.json"), 'w') as f:
        json.dump(kde_data, f)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores_label_1, shade=True, label='Label 1')
    sns.kdeplot(scores_label_0, shade=True, label='Label 0')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.title('KDE of Scores')
    plt.legend()
    plt.savefig(os.path.join(curves_save_path, "kde_curve.png"))
    #plt.savefig(os.path.join(curves_save_path, "kde_curve.pgf")) 
    plt.close()
    
    # Saving precision-recall data
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = average_precision_score(labels, scores)
    pr_data = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'average_precision': average_precision
    }
    with open(os.path.join(curves_save_path, "pr_data.json"), 'w') as f:
        json.dump(pr_data, f)

    plt.figure()
    plt.step(recall, precision, where='post', color='b', alpha=0.5, label=f'AP={average_precision:.5f}')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(curves_save_path, "aupr_curve.png"))
    #plt.savefig(os.path.join(curves_save_path, "aupr_curve.pgf")) 
    plt.close()

    # Saving ROC data
    fpr, tpr, thresholds = roc_curve(labels, scores)
    target_tpr = 0.95
    closest_tpr_index = np.argmin(np.abs(tpr - target_tpr))
    fpr95 = fpr[closest_tpr_index]
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'fpr95': fpr95
    }
    with open(os.path.join(curves_save_path, "roc_data.json"), 'w') as f:
        json.dump(roc_data, f)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve')
    plt.plot(fpr95, target_tpr, 'xr', markersize=10, label=f'FPR at TPR 95%: {fpr95:.5f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(curves_save_path, "roc_curve.png"))
    #plt.savefig(os.path.join(curves_save_path, "roc_curve.pgf")) 
    plt.close()
    print(f"FPR at 95% TPR: {fpr95}")

def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    tmp1 = (vec == 0) * 1
    tmp = np.diff(tmp1)
    edges, = np.nonzero(tmp)
    edge_vec = [edges + 1]

    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def save_evaluation_curves(scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    scores = scores.flatten()
    labels = labels.flatten()

    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        scores_each_video[video_id] = scores[start_idx:start_idx + video_frame_nums[video_id]]
        scores_each_video[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=17)
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    # Optionally, save the scores to JSON or NPY here before generating the plots
    save_scores_as_json(scores_each_video, curves_save_path)

    truth = []
    preds = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=1)
    auroc = auc(fpr, tpr)

        # Saving ROC data
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auroc': auroc
    }
    with open(os.path.join(curves_save_path, "roc_data.json"), 'w') as f:
        json.dump(roc_data, f)



    # draw ROC figure
    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    for i in sorted(scores_each_video.keys()):
        plt.figure()

        x = range(0, len(scores_each_video[i]))
        plt.xlim([x[0], x[-1] + 5])

        # anomaly scores
        plt.plot(x, scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

        # abnormal sections
        lb_one_intervals = nonzero_intervals(labels_each_video[i])
        for idx, (start, end) in enumerate(lb_one_intervals):
            plt.axvspan(start, end, alpha=0.5, color='red',
                        label="_" * idx + "Anomaly Intervals")

        plt.xlabel('Frames Sequence')
        plt.title('Test video %d' % (i + 1))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
        plt.close()

    return auroc


def save_scores_as_json(scores_each_video, curves_save_path):
    for video_id, scores in scores_each_video.items():
        file_path = os.path.join(curves_save_path, f"anomaly_scores_{video_id}.json")
        with open(file_path, 'w') as json_file:
            json.dump(scores.tolist(), json_file)

# This function converts a two-channel flow image to RGB color-coded image
def flow_to_color(flow):
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[0], flow[1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

def save_flow_images(flow_pred, flow_target, start_idx, frame_number, config):
    num_images = len(flow_pred)
    
    for i in range(num_images):
        pred_flow = flow_pred[i].cpu().detach().numpy()
        target_flow = flow_target[i].cpu().detach().numpy()

        def flow_to_color(flow):
            hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
            hsv[..., 1] = 255

            mag, ang = cv2.cartToPolar(flow[0], flow[1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return bgr
        
        pred_color = flow_to_color(pred_flow)
        target_color = flow_to_color(target_flow)
        
        pred_image = ToPILImage()(pred_color)
        target_image = ToPILImage()(target_color)

        pred_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Prediction_Flow', f'{frame_number}_{start_idx + i}.png')
        target_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Target_Flow', f'{frame_number}_{start_idx + i}.png')

        os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(target_save_path), exist_ok=True)

        pred_image.save(pred_save_path)
        target_image.save(target_save_path)
        
def load_and_resize_gt_images(gt_paths, target_size, cache_path='pixel_gt_dict.pickle'):
    if os.path.exists(cache_path):
        print("Loading from cache.")
        with open(cache_path, 'rb') as cache_file:
            pixel_gt_dict = pickle.load(cache_file)
        return pixel_gt_dict
    
    print("Create Pixel_gt_dict")
    pixel_gt_dict = {}
    for frame_number, gt_path in enumerate(gt_paths):
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)        
        resized_image = cv2.resize(gt_image, target_size, interpolation=cv2.INTER_AREA)
        for y in range(resized_image.shape[0]):
            for x in range(resized_image.shape[1]):
                is_anomaly = 1 if resized_image[y, x] == 255 else 0
                pixel_gt_dict[(frame_number, y, x)] = is_anomaly
    

    with open(cache_path, 'wb') as cache_file:
            pickle.dump(pixel_gt_dict, cache_file)
        
    return pixel_gt_dict

def robust_scale(array):
    median = np.median(array)
    q75, q25 = np.quantile(array, 0.75), np.quantile(array, 0.25)
    iqr = q75 - q25
    return (array - median) / iqr

def save_images_with_diff(grouped_preds, grouped_targets, start_idx,frame_number, config):
    num_images = len(grouped_preds)
    
    for i in range(num_images):
        pred = grouped_preds[i]
        target = grouped_targets[i]

        # Ensure tensors are correctly shaped for difference calculation
        if len(pred.shape) < 3:
            pred = pred.unsqueeze(0)
        if len(target.shape) < 3:
            target = target.unsqueeze(0)

        difference = torch.abs(pred - target).sum(dim=0)
        difference = difference / difference.max()

        to_pil = ToPILImage()
        pred_image = to_pil(pred.cpu().detach())
        target_image = to_pil(target.cpu().detach())

        pred_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Prediction_Frame', f'{frame_number}_{start_idx + i}.png')
        target_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Target_Frame', f'{frame_number}_{start_idx + i}.png')
        diff_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Difference_Frame', f'{frame_number}_{start_idx + i}.png')


        os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(target_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(diff_save_path), exist_ok=True)

        # Save prediction and target images
        pred_image.save(pred_save_path)
        target_image.save(target_save_path)

        # Generate and save the difference heatmap
        plt.figure(figsize=(pred_image.width / 100, pred_image.height / 100), dpi=100)
        ax = plt.gca()
        sns.heatmap(difference.cpu().numpy(), ax=ax, cmap='viridis', cbar=False)
        ax.set_axis_off()
        plt.savefig(diff_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        


def save_frame_plots(frame_number, pixel_scores, pixel_gt_dict, target_size, config):
    save_pixel_data_as_numpy(frame_number, pixel_scores,pixel_gt_dict, target_size, config)
    scores_image = np.zeros(target_size[::-1])
    gt_image = np.zeros(target_size[::-1])

    for y in range(target_size[1]):
        for x in range(target_size[0]):
            pixel_key = (frame_number, y, x)
            scores_image[y, x] = pixel_scores.get(pixel_key, 0)
            gt_image[y, x] = pixel_gt_dict.get(pixel_key, 0)

    
    pred_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Prediction_Pixel_Frame_inferno', f'frame_{frame_number}.png')
    target_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Target_Pixel_Frame_inferno', f'frame_{frame_number}.png')

    os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)

    # Visualize and save the anomaly scores image
    plt.figure()
    plt.imshow(scores_image, cmap='inferno')
    plt.axis('off')
    plt.savefig(pred_save_path, bbox_inches='tight')
    plt.close()

    # Visualize and save the ground truth image
    plt.figure()
    plt.imshow(gt_image, cmap='inferno')
    plt.axis('off')
    plt.savefig(target_save_path, bbox_inches='tight')
    plt.close()

def save_pixel_data_as_numpy(frame_number, pixel_scores, pixel_gt_dict, target_size, config):
    scores_array = np.zeros(target_size[::-1])
    gt_array = np.zeros(target_size[::-1])
    
    for y in range(target_size[1]):
        for x in range(target_size[0]):
            scores_key = (frame_number, y, x)
            gt_key = (frame_number, y, x)
            scores_array[y, x] = pixel_scores.get(scores_key, 0)
            gt_array[y, x] = pixel_gt_dict.get(gt_key, 0)
    
    scores_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Pixel_Scores_Numpy', f'pixel_scores_frame_{frame_number}.npy')
    gt_save_path = os.path.join(config["eval_root"], config["exp_name"], 'Pixel_GT_Numpy', f'pixel_gt_frame_{frame_number}.npy')
    
    os.makedirs(os.path.dirname(scores_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
    
    np.save(scores_save_path, scores_array)
    np.save(gt_save_path, gt_array)
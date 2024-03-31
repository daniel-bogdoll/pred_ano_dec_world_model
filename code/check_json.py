import pickle
import os
import numpy as np
import json

json_file_path = 'carla_data/ground_truth_demo/gt_label.json'
with open(json_file_path, 'r') as file:
    gt_labels = json.load(file)
gt_concat = np.concatenate(list(gt_labels.values()), axis=0)

print(len(gt_concat))


#gt_concat = np.concatenate(list(gt.values()), axis=0)


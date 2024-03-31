import os
import cv2
import numpy as np
from tqdm import tqdm

def find_bounding_boxes(segmentation, label):
    mask = segmentation == label
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    return bboxes

def visualize_bounding_boxes(segmentation, bboxes):
    # Convert the single-channel segmentation mask to a 3-channel image for visualization
    display_image = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Segmentation with Bounding Boxes', display_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    
def process_image(file_path, label):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    segmentation = image[:, :, 3]
    bboxes = find_bounding_boxes(segmentation, label)
    modified_bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]
    visualize_bounding_boxes(segmentation, modified_bboxes)
    return np.array(modified_bboxes)

def process_directory(directory, label):
    image_bboxes_list = []
    for filename in tqdm(sorted(os.listdir(directory)), desc="Processing images", leave=False):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            bboxes = process_image(file_path, label)
            image_bboxes_list.append(bboxes)
    return image_bboxes_list

# Update the root directory and output path as needed
root_dir = '/disk/vanishing_data/qw825/carla_dataset_small/trainval/train/'
label = 10

all_bboxes = []
for town in tqdm(sorted(os.listdir(root_dir)), desc="Processing towns"):
    town_path = os.path.join(root_dir, town)
    for scenario in tqdm(sorted(os.listdir(town_path)), desc=f"Processing scenarios in {town}", leave=False):
        scenario_path = os.path.join(town_path, scenario, 'depth_semantic')
        if os.path.isdir(scenario_path):
            image_bboxes_list = process_directory(scenario_path, label)
            all_bboxes.extend(image_bboxes_list)

print(f"Bounding boxes saved, Len:", len(all_bboxes))
npy_output_path = '/disk/vanishing_data/hb344/carla_test_custom_bboxes_1.npy'

np.save(npy_output_path, all_bboxes)

print(f"Bounding boxes saved to {npy_output_path}")

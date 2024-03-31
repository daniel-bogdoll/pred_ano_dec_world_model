import numpy as np
import cv2
import os
from tqdm import tqdm
import glob

def find_vehicle_bounding_boxes(image, semantic_ids=[14, 15]):
    blue_channel, green_channel, red_channel = cv2.split(image)
    unique_id = green_channel + (blue_channel << 8)
    
    semantic_mask = np.isin(red_channel, semantic_ids)
    
    filtered_unique_ids = unique_id * semantic_mask
    unique_vehicle_ids = np.unique(filtered_unique_ids[filtered_unique_ids > 0])
    bboxes_dict = {}
    for vehicle_id in unique_vehicle_ids:
        vehicle_mask = (filtered_unique_ids == vehicle_id)
        contours, _ = cv2.findContours(vehicle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(contour) for contour in contours]
        modified_bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]
        bboxes_dict[vehicle_id] = modified_bboxes
    return bboxes_dict

def filter_bounding_boxes(bboxes, image_shape, min_area=100, exclude_bottom_center_ratio=0.2, exclude_width_ratio=0.9):
    filtered_bboxes = []
    image_height, image_width = image_shape[:2]
    
    # Define the width range to exclude from the bottom center of the image
    exclude_width_start = image_width * (1 - exclude_width_ratio) / 2
    exclude_width_end = image_width * (1 + exclude_width_ratio) / 2

    # Define the height range to exclude from the bottom of the image
    exclude_height_start = image_height * (1 - exclude_bottom_center_ratio)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

        # Check if the bbox is in the bottom center region to be excluded
        if (y_min >= exclude_height_start or y_max >= exclude_height_start) and \
           (x_min >= exclude_width_start and x_max <= exclude_width_end):
            continue

        # Calculate the area of the bbox
        bbox_area = (x_max - x_min) * (y_max - y_min)

        # Keep the bbox only if it's larger than the minimum area
        if bbox_area >= min_area:
            filtered_bboxes.append(bbox)

    return filtered_bboxes



def process_image(file_path, visualize=True, min_area=100, exclude_bottom_ratio=0.2):
    image = cv2.imread(file_path)
    bboxes_dict = find_vehicle_bounding_boxes(image)
    # Erzeuge eine flache Liste aller Bounding Boxes
    flat_bboxes = [bbox for bbox_list in bboxes_dict.values() for bbox in bbox_list]

    # Filtere die Bounding Boxes
    filtered_bboxes = filter_bounding_boxes(flat_bboxes, image.shape, min_area, exclude_bottom_ratio)

    if visualize:
        display_image = visualize_bounding_boxes(image, filtered_bboxes)
        cv2.imshow('Vehicle Detection', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return filtered_bboxes


def visualize_bounding_boxes(image, bboxes):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

def process_directory(base_directory):
    image_bboxes_list = []
    counter = 0  # Initialize counter to keep track of every 50th image

    for scenario in tqdm(sorted(os.listdir(base_directory)), desc="Processing scenarios", leave=False):
        pattern = "INSTANCE-CAM*"
        scenario_dir = os.path.join(base_directory, scenario)
        matching_dirs = glob.glob(os.path.join(scenario_dir, pattern))
        
        if matching_dirs:
            scenario_path = matching_dirs[0]  # Use the first matching directory
            
            for filename in sorted(os.listdir(scenario_path)):
                if filename.endswith('.png'):
                    counter += 1
                    file_path = os.path.join(scenario_path, filename)
                    
                    # Set visualize to True when the counter is at a multiple of 50
                    flat_bboxes = process_image(file_path, visualize=(counter % 50 == 0))
                    image_bboxes_list.append(np.array(flat_bboxes))
            print(len(image_bboxes_list))
        else:
            print("No matching directories found.")

    return image_bboxes_list

    
# Pfad zum Basisverzeichnis mit den Szenarien
base_directory = '/disk/vanishing_data/hb344/testing_2/'

# Verarbeite alle Szenarien und speichere die Bounding Boxes
all_bboxes = process_directory(base_directory)

print(len(all_bboxes))
# Speichern des Objekt-Arrays
np.save("carla_2nd_test_boxes", all_bboxes)
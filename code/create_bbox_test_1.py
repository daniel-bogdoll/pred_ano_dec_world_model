import os
import cv2
import numpy as np
from tqdm import tqdm

def find_bounding_boxes(segmentation, label, min_area=50):
    mask = segmentation == label
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with an area smaller than min_area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    bboxes = [cv2.boundingRect(contour) for contour in filtered_contours]
    return bboxes

def visualize_bounding_boxes(image, bboxes):
    for bbox in bboxes:
        # Unpack the bounding box
        x_min, y_min, x_max, y_max = bbox
        # Draw a rectangle on the image using the modified bbox format
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # Display the image with bounding boxes
    cv2.imshow('Segmentation with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(file_path, labels, visualize=0):
    # Load the image
    image = cv2.imread(file_path)
    # Extract individual channels
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    all_bboxes = []
    for label in labels:  # Process each label
        # Create a mask where the R and G channels are 0, and the B channel is the label value
        mask = (red_channel == 0) & (green_channel == 0) & (blue_channel == label)
        
        # Use the mask to find contours and bounding boxes
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 50]
        bboxes = [cv2.boundingRect(contour) for contour in filtered_contours]

        # Convert bboxes to [x_min, y_min, x_max, y_max] format and extend the all_bboxes list
        modified_bboxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        all_bboxes.extend(modified_bboxes)
    
    # Create a copy of the original image for visualization
    display_image = image.copy()
    # Draw bounding boxes on the image copy
    if visualize:
        visualize_bounding_boxes(display_image, all_bboxes)
    
    return np.array(all_bboxes)


def process_directory(directory, label, channel=2):
    image_bboxes_list = []
    image_count = 0  # Initialize a counter for the images
    for filename in tqdm(sorted(os.listdir(directory)), desc="Processing images", leave=False):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            bboxes = process_image(file_path, labels, visualize=(image_count % 10 == 0))  # Pass the visualize flag
            if bboxes.size > 0:
                image_bboxes_list.append(bboxes)
            image_count += 1  # Increment the image counter
    return image_bboxes_list


# Update the root directory and output path as needed
root_dir = '/disk/vanishing_data/hb344/testing_1/'
labels = [70, 142]   # Update the label as needed

all_bboxes = []
for town in tqdm(sorted(os.listdir(root_dir)), desc="Processing towns"):
    town_path = os.path.join(root_dir, town)
    scenario_path = os.path.join(town_path, 'SEMANTIC_IMG')
    if os.path.isdir(scenario_path):
        image_bboxes_list = process_directory(scenario_path, labels)
        all_bboxes.extend(image_bboxes_list)

print(f"Bounding boxes saved, Len: {len(all_bboxes)}")
npy_output_path = '/disk/vanishing_data/hb344/carla_test_custom_bboxes_1.npy'

# Save the bounding boxes to an .npy file
np.save(npy_output_path, all_bboxes)

print(f"Bounding boxes saved to {npy_output_path}")

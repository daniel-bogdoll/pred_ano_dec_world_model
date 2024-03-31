import os
import cv2
import numpy as np

def visualize_bounding_boxes(image, bboxes):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bboxes_npy_path = 'carla_data/bboxes_carla/carla_train_custom_bboxes.npy'
all_bboxes = np.load(bboxes_npy_path, allow_pickle=True)

image_dir = '/disk/vanishing_data/qw825/carla_dataset_small/trainval/train/Town01/0000/image/'

# Make sure to sort the list of filenames so they match the order of bounding boxes
image_files = sorted(os.listdir(image_dir))

for idx, filename in enumerate(image_files.json):
    if filename.endswith('.png'):
        file_path = os.path.join(image_dir, filename)
        image = cv2.imread(file_path)
        # Retrieve the corresponding bounding boxes
        image_bboxes = all_bboxes[idx]
        visualize_bounding_boxes(image, image_bboxes)

# Make sure the paths and file extensions are correct for your specific case

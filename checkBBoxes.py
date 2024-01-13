import numpy as np
import cv2
import os

def draw_bboxes_on_image(image, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def load_images_with_names_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):  
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)  # Dateinamen speichern
    return images, filenames

bbox_file_path = '/fzi/ids/hb344/Desktop/hf2vad/data/ped21313/ped2_bboxes_test.npy'
base_images_folder_path = 'data/ped21313/testing/frames'

bboxes = np.load(bbox_file_path, allow_pickle=True)

for folder_name in sorted(os.listdir(base_images_folder_path)):
    current_folder_path = os.path.join(base_images_folder_path, folder_name)
    print(f"Verarbeitung von: {current_folder_path}")
    
    images, filenames = load_images_with_names_from_folder(current_folder_path)
    
    
    for img, bbox, filename in zip(images, bboxes, filenames):
        img_with_bbox = draw_bboxes_on_image(img, bbox)
        cv2.imshow(f"Image with Bounding Box - {filename}", img_with_bbox)
        print(f"Anzeige von: {filename}") 
        cv2.waitKey(0)  
    
    cv2.destroyAllWindows()
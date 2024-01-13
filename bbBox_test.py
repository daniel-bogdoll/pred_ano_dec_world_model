import cv2
import numpy as np

image = cv2.imread('ref_sensors_semantic.jpg')

pedestrian_color = (220, 20, 60)
vehicle_color = (0, 0, 142)
pedestrian_mask = cv2.inRange(image, pedestrian_color, pedestrian_color)
vehicle_mask = cv2.inRange(image, vehicle_color, vehicle_color)

pedestrian_contours, _ = cv2.findContours(pedestrian_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vehicle_contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = []

def add_bounding_boxes(contours, label, color):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append({'label': label, 'box': [x, y, x+w, y+h]})
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

add_bounding_boxes(pedestrian_contours, 'pedestrian', (255, 0, 0)) # Blaue Boxen für Fußgänger
add_bounding_boxes(vehicle_contours, 'vehicle', (0, 255, 0)) # Grüne Boxen für Fahrzeuge

np.save('/mnt/data/bounding_boxes.npy', bounding_boxes)

cv2.imwrite('/mnt/data/image_with_bboxes.png', image)

np.save('bounding_boxes.npy', bounding_boxes)

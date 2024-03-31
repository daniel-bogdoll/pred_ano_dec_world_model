import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '/disk/users/hb344/no_backup/Desktop/hf2vad/data/carla_local/semantic/trainval/train/Town04/0010/depth_semantic/depth_semantic_000000459.png'

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
height, width = image.shape[:2]  # Die Reihenfolge von Höhe und Breite ist hier vertauscht

print("Breite:", width)
print("Höhe:", height)


channel_1, channel_2, channel_3, channel_4 = cv2.split(image)

plt.imshow(channel_4)
plt.show()

semantic_channel = image[:, :, 3]
binary_image = np.where(semantic_channel == 10, 255, 0).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8) 
#remove small black spaces
binary_image_closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Konturen finden
contours, _ = cv2.findContours(binary_image_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

binary_image_colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
for contour in contours:
    if cv2.contourArea(contour) > 50:  
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(binary_image_colored, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Ergebnisbild anzeigen oder speichern
cv2.imshow('Bounding Boxes on Binary Image', binary_image_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()

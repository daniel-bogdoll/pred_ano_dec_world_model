import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

# Load the bounding boxes
stc_file = 'data/ped2/training/chunked_samples/chunked_samples_00.pkl'
stc_data = joblib.load(stc_file)

# Example: Display the predicted frame for the first STC
stc_index = 0  # Index of the STC to display
predicted_frame_index = stc_data['pred_frame'] # Last frame index in the selected STC
print(predicted_frame_index)
# Load the corresponding frame image
# Adjust this path and format according to how your frames are stored
frame_image_path = f'data/ped2/training/frames/Train001/{predicted_frame_index:03d}.tif'
frame_image = cv2.imread(frame_image_path)

if frame_image is None:
    raise ValueError(f"Could not load the frame image at {frame_image_path}")

# Display the frame
plt.imshow(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Frame: {predicted_frame_index}")
plt.show()

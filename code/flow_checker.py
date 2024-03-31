import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualize_optical_flow(flow):
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Normalize the magnitude to range [0, 1]
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    # Scale the normalized magnitude to range [0, 255] and convert to 8-bit
    magnitude = (magnitude * 255).astype(np.uint8)
    # Map the angles to hue values in range [0, 180] for HSV color space
    hsv_hue = angle * (180 / np.pi / 2)
    # Create an HSV representation of the flow
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hsv_hue
    hsv[..., 1] = 255  # Set saturation to maximum
    hsv[..., 2] = magnitude
    # Convert HSV to RGB for visualization
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

# Loop through each image index
#for i in range(201):  # assuming you have 1501 images, from 0000 to 1500
#for i in range(3520, 12312, 2):
    #image_number = str(i).zfill(9)  # zero-pad the index to 12 digits
    #image_path = f'/disk/vanishing_data/qw825/carla_dataset_small/trainval/train/Town01/0000/image/image_{image_number}.png'
    #flow_path = f'carla_data/flows_carla/training/flows/image/image_{image_number}.png.npy'
    image_path = f'/disk/users/hb344/no_backup/Desktop/hf2vad/data/carla_local/testing/0005/RGB_IMG/RGB_IMG_{i}.png'
    flow_path = f'/disk/users/hb344/no_backup/Desktop/hf2vad/carla_data/flows_carla/testing/testing/0005/RGB_IMG/RGB_IMG_{i}.png.npy'
    
    # Load the image and the optical flow data
    image = cv2.imread(image_path)
    flow = np.load(flow_path)

    # Check if the image is loaded properly
    if image is None:
        raise ValueError(f"The image at {image_path} could not be loaded. Please check the file path and ensure it is correct.")

    # Convert the flow data to color-coded representation
    color_coded_flow = visualize_optical_flow(flow)

    # Display the image and the optical flow
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image {i}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(color_coded_flow)
    plt.title(f'Optical Flow {i}')
    plt.axis('off')

    plt.show()
    # Optionally, you could save the figure using plt.savefig if required

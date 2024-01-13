from clearml import Dataset
import matplotlib.pyplot as plt
import os

ds = Dataset.get(dataset_id='f696541616f44f36b44c4f99f33e6172', dataset_version='None')

dataset_path = ds.get_local_copy()

image_paths = [
    "trainval/train/Town01/0000/image/image_000000299.png",
]

for img_path in image_paths:
    full_path = os.path.join(dataset_path, img_path)
    image = plt.imread(full_path)
    plt.imshow(image)
    plt.show()


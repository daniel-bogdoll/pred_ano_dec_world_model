import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

def visualize_sample_on_frame(frame, stc_patch, bbox):
    y1, x1, y2, x2 = map(int, bbox)

    # Stelle sicher, dass die Bounding-Box innerhalb des Frames liegt
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)

    # Überprüfe, ob die Bounding-Box gültige Dimensionen hat
    if y2 <= y1 or x2 <= x1:
        print(f"Ungültige Bounding-Box-Dimensionen. Überspringe: {bbox}")
        return None

    # Skaliere das stc_patch auf die Größe der Bounding-Box
    stc_patch_resized = cv2.resize(stc_patch, (x2 - x1, y2 - y1))

    frame_with_stc = frame.copy()
    frame_with_stc[y1:y2, x1:x2] = stc_patch_resized
    cv2.rectangle(frame_with_stc, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame_with_stc


#stc_file = '/fzi/ids/hb344/no_backup/Desktop/hf2vad/data/ped2/training/chunked_samples/chunked_samples_00.pkl'
stc_file = 'carla_data/STC/testing/chunked_samples/chunked_samples_00.pkl'

frame_paths = ['data/ped2/training/frames/Train001/{:03d}.tif'.format(i) for i in range(1, 201)]
frame_paths = [f'data/carla_local/testing/0001/ground-truth/SEMANTIC_IMG_{i}.png' for i in range(403, 600, 2)]

stc_data = joblib.load(stc_file)

for sample_id in range(len(stc_data['sample_id'])):
    stc_patch = stc_data['appearance'][sample_id][-1]
    bbox = stc_data['bbox'][sample_id]
    frame_idx = stc_data['pred_frame'][sample_id][-1]

    frame_path = frame_paths[frame_idx]
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warnung: Frame bei {frame_path} konnte nicht geladen werden.")
        continue

    visualized_frame = visualize_sample_on_frame(frame, stc_patch, bbox)
    if visualized_frame is not None:
        plt.imshow(cv2.cvtColor(visualized_frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Sample ID: {sample_id}, Frame: {frame_idx}")
        plt.show()


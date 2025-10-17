# train.py
import os
import cv2
import numpy as np
import pickle

def main(dataset_dir='dataset', trainer_path='trainer.yml', labels_path='labels.pickle'):
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset folder '{dataset_dir}' not found. Run collect_faces.py first.")
        return

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[current_label] = person_name
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200,200))
            faces.append(img)
            labels.append(current_label)
        print(f"[INFO] Loaded images for '{person_name}' -> label {current_label}")
        current_label += 1

    if len(faces) == 0:
        print("[ERROR] No images found. Exiting.")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("[ERROR] cv2.face.LBPHFaceRecognizer_create not found. Install opencv-contrib-python.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.write(trainer_path)
    with open(labels_path, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"[INFO] Training complete. Saved trainer to '{trainer_path}' and labels to '{labels_path}'.")

if __name__ == "__main__":
    main()

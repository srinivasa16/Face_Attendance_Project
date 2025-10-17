# recognize.py
import cv2
import os
import pickle
import csv
from datetime import datetime

def mark_attendance(name, csv_path='attendance.csv'):
    file_exists = os.path.isfile(csv_path)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time'])
        writer.writerow([name, date_str, time_str])

def main(trainer_path='trainer.yml', labels_path='labels.pickle', threshold=80):
    if not os.path.isfile(trainer_path) or not os.path.isfile(labels_path):
        print("[ERROR] trainer.yml or labels.pickle not found. Run train.py first.")
        return
    with open(labels_path, 'rb') as f:
        label_map = pickle.load(f)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    seen = set()
    print("[INFO] Starting recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200,200))
            label, confidence = recognizer.predict(face)  # lower = better
            name = "Unknown"
            if confidence < threshold:
                name = label_map.get(label, "Unknown")
                if name not in seen:
                    seen.add(name)
                    mark_attendance(name)
                    print(f"[INFO] Marked attendance for {name} (conf={confidence:.2f})")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} {int(confidence)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exiting. Attendance saved to attendance.csv")

if __name__ == "__main__":
    main()

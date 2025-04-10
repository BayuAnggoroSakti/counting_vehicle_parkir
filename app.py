import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# Load model YOLOv8
model = YOLO("yolov8m.pt")

# Warna bounding box
colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 255, 255),
    'motorbike': (255, 0, 255)
}

st.set_page_config(layout="wide")
st.title("🚗 Deteksi Kendaraan Parkir dengan YOLOv8")

uploaded_file = st.file_uploader("Unggah Video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi YOLO
        results = model(frame, verbose=False)

        # Inisialisasi counter
        counts = {label: 0 for label in colors.keys()}

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in counts:
                    counts[label] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = colors[label]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tampilkan count
        y_offset = 30
        for label, count in counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[label], 2)
            y_offset += 30

        # Convert BGR ke RGB untuk Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()

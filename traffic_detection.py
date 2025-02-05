import cv2
import numpy as np

# Load YOLO model
weights_path = "yolov4.weights"  
config_path = "yolov4.cfg"       
names_path = "coco.names"        

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
video_path = "traffic.mp4"  # Ganti dengan video lalu lintas
cap = cv2.VideoCapture(video_path)

# Set area deteksi kemacetan (koordinat ROI)
ROI_top_left = (200, 300)
ROI_bottom_right = (1000, 700)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Process detections
    vehicles = ["car", "bus", "truck", "motorbike"]
    vehicle_count = 0

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in vehicles:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Cek apakah kendaraan ada dalam ROI
                if ROI_top_left[0] < center_x < ROI_bottom_right[0] and ROI_top_left[1] < center_y < ROI_bottom_right[1]:
                    vehicle_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tentukan tingkat kemacetan
    congestion_status = "Low"
    if vehicle_count > 10:
        congestion_status = "High"
    elif vehicle_count > 5:
        congestion_status = "Medium"

    # Gambar ROI dan status kemacetan
    cv2.rectangle(frame, ROI_top_left, ROI_bottom_right, (0, 0, 255), 2)
    cv2.putText(frame, f"Traffic: {congestion_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Traffic Congestion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

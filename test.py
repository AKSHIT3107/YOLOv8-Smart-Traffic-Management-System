from ultralytics import YOLO
import cv2
import time

# Load trained YOLOv8 model
model = YOLO("smart_traffic/emergency_vehicle_model25/weights/best.pt")

# Define emergency classes
emergency_classes = ["ambulance", "police", "fire_truck"]

# Start webcam
cap = cv2.VideoCapture(0)

# Prevent spamming (1 message every N seconds)
last_alert_time = 0
alert_interval = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Draw bounding box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Check if emergency vehicle is detected
            if label in emergency_classes and conf > 0.8:
                current_time = time.time()
                if current_time - last_alert_time > alert_interval:
                    print(f"🚨 EMERGENCY VEHICLE DETECTED: {label.upper()}")
                    print("📩 Sending message to authority: TURN TRAFFIC LIGHT GREEN 🚦")
                    # TODO: Add actual code to send signal/message via:
                    # - API request (requests.post)
                    # - GPIO control
                    # - SMS / IFTTT / MQTT
                    last_alert_time = current_time

    # Show result
    cv2.imshow("Emergency Vehicle Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

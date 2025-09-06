from ultralytics import YOLO
import cv2
import time

# Load your trained YOLOv8 model
model = YOLO("smart_traffic/emergency_vehicle_model25/weights/best.pt")

# Class names (from your data.yaml)
vehicle_classes = ["ambulance", "car", "fire_truck", "police"]
emergency_classes = {"ambulance", "police", "fire_truck"}

# Start webcam
cap = cv2.VideoCapture(0)

# Timers
last_emergency_time = 0
emergency_override_duration = 10  # seconds to override signal
alert_interval = 10               # time between alerts

# Default green light duration
green_light_duration = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    emergency_detected = False
    vehicle_count = 0

    # Run detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Skip low confidence
            if conf < 0.5:
                continue

            # Draw box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Count all vehicles
            if label in vehicle_classes:
                vehicle_count += 1

            # Check emergency
            if label in emergency_classes and conf > 0.8:
                emergency_detected = True

    # Emergency logic
    if emergency_detected:
        if current_time - last_emergency_time > alert_interval:
            print("🚨 EMERGENCY VEHICLE DETECTED — SIGNAL OVERRIDE ACTIVE")
            print("📩 Turning GREEN for emergency vehicle")
            last_emergency_time = current_time
        signal_status = "GREEN (Emergency Override)"
        green_light_duration = emergency_override_duration
    else:
        # Traffic density logic
        if vehicle_count > 10:
            green_light_duration = 10
        elif vehicle_count > 5:
            green_light_duration = 7
        else:
            green_light_duration = 5
        signal_status = f"GREEN for {green_light_duration}s"

    # Display information
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Signal: {signal_status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Smart Traffic System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
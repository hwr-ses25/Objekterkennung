import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Sprachmodul initialisieren
engine = pyttsx3.init()

# YOLOv8n Modell laden
model = YOLO('yolov8n.pt')

# Wähle Kamera: 0 = Mac-Webcam, 1 oder 2 = USB-/externe Kamera (z. B. NiklAVision)
camera_index = 1  # ← hier ggf. 0, 1 oder 2 ausprobieren
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"❌ Kamera {camera_index} konnte nicht geöffnet werden.")
    exit()

# Sprech-Kontrolle
last_spoken = {}
speak_interval = 5  # Sekunden

print("✅ Kamera läuft – drücke 'q' zum Beenden")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kein Kamerabild.")
        break

    results = model(frame)[0]
    annotated_frame = frame.copy()
    current_time = time.time()
    detected_now = set()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detected_now.add(label)

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for name in detected_now:
        # Nur sprechen, wenn neu oder lange her
        if name not in last_spoken or (current_time - last_spoken[name]) > speak_interval:
            print(f"🔎 Erkannt: {name}")
            engine.say(name)
            engine.runAndWait()
            last_spoken[name] = current_time

    cv2.imshow("YOLOv8 Objekterkennung", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

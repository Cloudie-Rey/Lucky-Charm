import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = results[0].names[int(box.cls[0])]
        
        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 212), 2)
            cv2.putText(frame, f"{label} {conf:.0%}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 0, 212), 2)
    
    cv2.imshow("Lucky Charm", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
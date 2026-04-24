from ultralytics import YOLO
import cv2
import csv

VIDEO_PATH = "08fd33_4.mp4"                
OUTPUT_VIDEO = "annotated_players_ball.mp4"
OUTPUT_CSV = "detections_players_ball.csv"

PERSON_CONF = 0.40       
BALL_CONF = 0.30


model = YOLO("yolov8s.pt") 
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "label", "confidence", "x1", "y1", "x2", "y2", "track_id"])  # ### CHANGED: added track_id

frame_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_id += 1

    results = model.track(
        frame,
        conf=min(PERSON_CONF, BALL_CONF),
        iou=0.45,
        classes=[0, 32],              # 0=person, 32=sports ball (COCO)
        tracker="bytetrack.yaml",     # built-in ByteTrack defaults
        persist=True,                 # keep IDs across frames
        verbose=False
    )
    result = results[0]
    ids = result.boxes.id  # may be None for a frame

    # iterate detections
    for i, box in enumerate(result.boxes):   
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        # only keep players (person) and ball (sports ball)
        if label == "person" and conf < PERSON_CONF:
            continue
        if label == "sports ball" and conf < BALL_CONF:
            continue
        if label not in ("person", "sports ball"):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        tid = int(ids[i]) if (ids is not None and ids[i] is not None) else -1  # ### NEW

        color = (255, 0, 0) if label == "person" else (0,255, 0) 
        tag = f"{label} {conf:.2f}" + (f" ID:{tid}" if tid >= 0 else "")    
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, tag, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # log to CSV (now also write the track_id)
        csv_writer.writerow([frame_id, label, f"{conf:.4f}", x1, y1, x2, y2, tid])  # ### CHANGED

    writer.write(frame)

cap.release()
writer.release()
csv_file.close()

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("soccer-analysis")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        "supervision",
        "ultralytics", 
        "opencv-python",
        "matplotlib",
        "tqdm",
        "roboflow",
        "numpy",
        "transformers",
        "umap-learn",
        "scikit-learn",
        "sentencepiece",
        "protobuf",
        "torch",
        "torchvision",
    )
    .run_commands(
        "cd /root && git clone https://github.com/roboflow/sports.git",
        "cd /root/sports && pip install -e .",
    )
)

# Create a volume for storing videos and outputs
volume = modal.Volume.from_name("soccer-videos", create_if_missing=True)

VOLUME_PATH = "/data"


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 1 hour timeout
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("roboflow-api")],
)
def process_video(video_filename: str):
    """
    Process a soccer video with player tracking and pitch projection.
    
    Args:
        video_filename: Name of video file in the Modal volume
    """
    import cv2
    import numpy as np
    import tempfile
    from tqdm import tqdm
    import supervision as sv
    from sports.common.team import TeamClassifier
    from sports.common.view import ViewTransformer
    from sports.configs.soccer import SoccerPitchConfiguration
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    from roboflow import Roboflow
    
    # Configuration
    VIDEO_PATH = f"{VOLUME_PATH}/{video_filename}"
    base_name = Path(video_filename).stem
    OUT_PATH = f"{VOLUME_PATH}/{base_name}_annotated.mp4"
    OUT_PITCH = f"{VOLUME_PATH}/{base_name}_pitch.mp4"
    
    PASS1_SECONDS = 60
    STRIDE_FRAMES = 10
    MIN_CROPS = 300
    MIN_PLAYER_H_FRAC = 0.06
    MIN_AREA_FRAC = 0.0015
    
    PLAYER_CONF = 30
    BALL_CONF = 30
    NMS_THRESH = 0.5
    FIELD_CONF = 30
    KP_CONF_THRESH = 0.50
    MIN_KP_COUNT = 6
    H_UPDATE_EVERY = 5
    
    CONFIG = SoccerPitchConfiguration()
    PITCH_PADDING = 50
    PITCH_SCALE = 0.1
    
    TEAM0_ID = 0
    TEAM1_ID = 1
    REFEREE_ID = 2
    
    DEBUG_EVERY = 30
    DEBUG_KP_EVERY = 60
    
    # Initialize Roboflow
    rf_api_key = os.environ["ROBOFLOW_API_KEY"]
    rf = Roboflow(api_key=rf_api_key)
    
    player_model = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc").version(1).model
    ball_model = rf.workspace("roboflow-jvuqo").project("football-ball-detection-rejhg").version(1).model
    field_model = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi").version(1).model
    
    print("✓ Roboflow models loaded")
    
    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25, height=21, outline_thickness=1
    )
    
    # Helper functions
    def norm_conf(c):
        c = float(c)
        return c / 100.0 if c > 1.0 else c
    
    def clamp_xyxy(x1, y1, x2, y2, W, H):
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W - 1, int(x2)))
        y2 = max(0, min(H - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    
    def safe_crop(frame, xyxy):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        clamped = clamp_xyxy(x1, y1, x2, y2, W, H)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
    
    def detections_from_player_json(pred_json, W, H):
        raw = pred_json.get("predictions", [])
        boxes, confs = [], []
        for det in raw:
            x = float(det["x"]); y = float(det["y"])
            ww = float(det["width"]); hh = float(det["height"])
            c = norm_conf(det.get("confidence", 1.0))
            
            x1, y1 = x - ww/2.0, y - hh/2.0
            x2, y2 = x + ww/2.0, y + hh/2.0
            
            clamped = clamp_xyxy(x1, y1, x2, y2, W, H)
            if clamped is None:
                continue
            boxes.append(clamped)
            confs.append(c)
        
        if len(boxes) == 0:
            return sv.Detections.empty()
        
        dets = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.full((len(boxes),), TEAM0_ID, dtype=int)
        )
        return dets.with_nms(threshold=NMS_THRESH, class_agnostic=True)
    
    def ball_detection_from_json(ball_json, W, H):
        raw = ball_json.get("predictions", [])
        if len(raw) == 0:
            return sv.Detections.empty()
        
        raw = sorted(raw, key=lambda d: norm_conf(d.get("confidence", 1.0)), reverse=True)
        b = raw[0]
        
        bx, by = float(b["x"]), float(b["y"])
        bw, bh = float(b.get("width", 12.0)), float(b.get("height", 12.0))
        bc = norm_conf(b.get("confidence", 1.0))
        
        x1, y1 = bx - bw/2.0, by - bh/2.0
        x2, y2 = bx + bw/2.0, by + bh/2.0
        clamped = clamp_xyxy(x1, y1, x2, y2, W, H)
        if clamped is None:
            return sv.Detections.empty()
        
        det = sv.Detections(
            xyxy=np.array([clamped], dtype=np.float32),
            confidence=np.array([bc], dtype=np.float32),
            class_id=np.array([999], dtype=int)
        )
        det.xyxy = sv.pad_boxes(xyxy=det.xyxy, px=10)
        return det
    
    def good_player_box(box, W, H):
        x1, y1, x2, y2 = box
        bw, bh = float(x2 - x1), float(y2 - y1)
        area_frac = (bw * bh) / float(W * H)
        h_frac = bh / float(H)
        return (area_frac >= MIN_AREA_FRAC) and (h_frac >= MIN_PLAYER_H_FRAC)
    
    def jersey_crop(crop):
        if crop is None:
            return None
        h = crop.shape[0]
        return crop[: int(0.65 * h), :]
    
    def is_referee_crop(crop):
        if crop is None or crop.size == 0:
            return False
        small = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        v_mean, s_mean = float(np.mean(V)), float(np.mean(S))
        return (v_mean < 70) and (s_mean < 80)
    
    def parse_field_keypoints_from_json(field_json, config):
        preds = field_json.get("predictions", [])
        if not preds:
            return None, None
        
        N = len(config.vertices)
        xy = np.full((N, 2), np.nan, dtype=np.float32)
        conf = np.zeros((N,), dtype=np.float32)
        
        label_to_idx = {}
        if hasattr(config, "labels") and config.labels:
            label_to_idx = {str(lbl): i for i, lbl in enumerate(config.labels)}
        
        def resolve_idx(obj, N, label_to_idx):
            if "class" in obj and obj["class"] is not None:
                k = str(obj["class"])
                if k in label_to_idx:
                    return label_to_idx[k]
            
            cid = obj.get("class_id", obj.get("classId", None))
            if cid is None:
                return None
            try:
                cid = int(cid)
            except:
                return None
            
            if cid == 0:
                return 0
            if 0 <= cid < N:
                return cid
            if 1 <= cid <= N:
                return cid - 1
            return None
        
        def put(idx, x, y, c):
            if idx is None or not (0 <= idx < N):
                return
            c = norm_conf(c)
            if c >= conf[idx]:
                xy[idx, 0] = float(x)
                xy[idx, 1] = float(y)
                conf[idx] = float(c)
        
        found_any = False
        for det in preds:
            kps = det.get("keypoints", None) if isinstance(det, dict) else None
            if isinstance(kps, list) and len(kps) > 0:
                found_any = True
                for kp in kps:
                    if isinstance(kp, dict) and ("x" in kp) and ("y" in kp):
                        put(resolve_idx(kp, N, label_to_idx), kp["x"], kp["y"], kp.get("confidence", 0.0))
        
        if found_any:
            return xy, conf
        
        for kp in preds:
            if isinstance(kp, dict) and ("x" in kp) and ("y" in kp):
                put(resolve_idx(kp, N, label_to_idx), kp["x"], kp["y"], kp.get("confidence", 0.0))
        
        return xy, conf
    
    # PASS 1: Collect training crops
    print("\n=== PASS 1: Collecting training crops ===")
    team_classifier = TeamClassifier(device="cuda")
    train_crops = []
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0
    max_frames = int(PASS1_SECONDS * fps)
    
    frame_i = 0
    pbar = tqdm(total=max_frames, desc="Collecting crops")
    
    while frame_i < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_i % STRIDE_FRAMES != 0:
            frame_i += 1
            pbar.update(1)
            continue
        
        H, W = frame.shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, frame)
            player_preds = player_model.predict(tmp.name, confidence=PLAYER_CONF).json()
            dets = detections_from_player_json(player_preds, W, H)
            
            if len(dets) > 0:
                for box in dets.xyxy:
                    if not good_player_box(box, W, H):
                        continue
                    crop = jersey_crop(safe_crop(frame, box))
                    if crop is None:
                        continue
                    if is_referee_crop(crop):
                        continue
                    train_crops.append(crop)
        
        if len(train_crops) >= MIN_CROPS:
            frame_i += 1
            pbar.update(1)
            break
        
        frame_i += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"✓ Collected {len(train_crops)} training crops")
    
    if len(train_crops) < 80:
        print("⚠ Warning: Few crops collected. Results may vary.")
    
    print("Training team classifier...")
    team_classifier.fit(train_crops)
    print("✓ Team classifier trained")
    
    # PASS 2: Process full video
    print("\n=== PASS 2: Processing full video ===")
    tracker = sv.ByteTrack()
    tracker.reset()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W0, H0))
    
    pitch_base = draw_pitch(
        config=CONFIG, padding=PITCH_PADDING, scale=PITCH_SCALE,
        background_color=sv.Color.from_hex("#1E7A3A"),
        line_color=sv.Color.WHITE
    )
    PH, PW = pitch_base.shape[:2]
    out_pitch = cv2.VideoWriter(OUT_PITCH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (PW, PH))
    
    frame_to_pitch_transformer = None
    last_H_frame = None
    frame_i = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        H, W = frame.shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, frame)
            
            # Player detections
            player_preds = player_model.predict(tmp.name, confidence=PLAYER_CONF).json()
            dets = detections_from_player_json(player_preds, W, H)
            tracked = tracker.update_with_detections(detections=dets)
            
            if tracked is None or len(tracked) == 0:
                tracked = sv.Detections.empty()
            
            # Classify players
            det_is_ref, classify_crops, classify_idx = [], [], []
            if len(tracked) > 0:
                for i, box in enumerate(tracked.xyxy):
                    crop = jersey_crop(safe_crop(frame, box))
                    if crop is None:
                        det_is_ref.append(False)
                        continue
                    is_ref = is_referee_crop(crop)
                    det_is_ref.append(is_ref)
                    if not is_ref:
                        classify_crops.append(crop)
                        classify_idx.append(i)
            
            class_ids = np.full((len(tracked),), TEAM0_ID, dtype=int) if len(tracked) > 0 else np.array([], dtype=int)
            if len(classify_crops) > 0:
                preds = team_classifier.predict(classify_crops)
                for i, p in zip(classify_idx, preds):
                    class_ids[i] = TEAM0_ID if int(p) == 0 else TEAM1_ID
            
            if len(tracked) > 0 and len(det_is_ref) == len(tracked):
                for i, is_ref in enumerate(det_is_ref):
                    if is_ref:
                        class_ids[i] = REFEREE_ID
            if len(tracked) > 0:
                tracked.class_id = class_ids.astype(int)
            
            # Ball detection
            ball_preds = ball_model.predict(tmp.name, confidence=BALL_CONF).json()
            ball_det = ball_detection_from_json(ball_preds, W, H)
            
            # Annotate frame
            annotated = frame.copy()
            if len(tracked) > 0:
                annotated = ellipse_annotator.annotate(scene=annotated, detections=tracked)
                labels = [f"#{tid}" for tid in tracked.tracker_id]
                annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
            if len(ball_det) > 0:
                annotated = triangle_annotator.annotate(scene=annotated, detections=ball_det)
            
            # Update homography
            if (frame_i % H_UPDATE_EVERY == 0) or (frame_to_pitch_transformer is None):
                field_json = field_model.predict(tmp.name, confidence=FIELD_CONF).json()
                xy, conf = parse_field_keypoints_from_json(field_json, CONFIG)
                
                if (xy is None) or (conf is None) or (float(np.nanmax(conf)) == 0.0):
                    field_json = field_model.predict(tmp.name, confidence=FIELD_CONF / 100.0).json()
                    xy, conf = parse_field_keypoints_from_json(field_json, CONFIG)
                
                if xy is not None and conf is not None:
                    filt = (conf > KP_CONF_THRESH) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
                    good_n = int(filt.sum())
                    
                    if good_n >= MIN_KP_COUNT:
                        frame_reference_points = xy[filt].astype(np.float32)
                        pitch_reference_points = np.array(CONFIG.vertices, dtype=np.float32)[filt]
                        frame_to_pitch_transformer = ViewTransformer(
                            source=frame_reference_points,
                            target=pitch_reference_points
                        )
                        last_H_frame = frame_i
            
            # Pitch projection
            pitch_frame = pitch_base.copy()
            
            if frame_to_pitch_transformer is not None:
                if len(ball_det) > 0:
                    frame_ball_xy = ball_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
                    pitch_ball_xy = frame_to_pitch_transformer.transform_points(points=frame_ball_xy)
                    pitch_frame = draw_points_on_pitch(
                        config=CONFIG, xy=pitch_ball_xy,
                        face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                        radius=10, thickness=2, pitch=pitch_frame
                    )
                
                if len(tracked) > 0:
                    frame_players_xy = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
                    pitch_players_xy = frame_to_pitch_transformer.transform_points(points=frame_players_xy)
                    
                    for team_id, color in [(TEAM0_ID, "#00BFFF"), (TEAM1_ID, "#FF1493"), (REFEREE_ID, "#FFD700")]:
                        mask = (tracked.class_id == team_id)
                        if mask.any():
                            pitch_frame = draw_points_on_pitch(
                                config=CONFIG, xy=pitch_players_xy[mask],
                                face_color=sv.Color.from_hex(color),
                                edge_color=sv.Color.BLACK,
                                radius=16 if team_id != REFEREE_ID else 14,
                                thickness=2, pitch=pitch_frame
                            )
            
            out.write(annotated)
            out_pitch.write(pitch_frame)
        
        frame_i += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    out_pitch.release()
    
    # Commit volume changes
    volume.commit()
    
    print(f"\n✓ Processing complete!")
    print(f"  - Annotated video: {OUT_PATH}")
    print(f"  - Pitch view: {OUT_PITCH}")
    
    return {
        "annotated_path": OUT_PATH,
        "pitch_path": OUT_PITCH,
        "base_name": base_name
    }

@app.function(volumes={VOLUME_PATH: volume})
def upload_to_volume(filename: str, data: bytes):
    path = f"{VOLUME_PATH}/{filename}"
    with open(path, "wb") as f:
        f.write(data)

@app.function(volumes={VOLUME_PATH: volume})
def download_from_volume(filename: str) -> bytes:
    path = f"{VOLUME_PATH}/{filename}"
    with open(path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(video_path: str = "test_clip.mp4"):
    """
    Main entry point - uploads video, processes it, and downloads results.
    
    Args:
        video_path: Path to input video file on your local machine
    """
    import shutil
    from pathlib import Path
    
    # Create local outputs directory
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Uploading video: {video_path}")
    
    # Upload video to Modal volume
    video_filename = Path(video_path).name
    
    # Check if file exists locally
    video_file = Path(video_path)

    # If not an absolute path, try relative to current directory
    if not video_file.is_absolute():
        video_file = Path.cwd() / video_path

    if not video_file.exists():
        print(f" Error: Video file not found: {video_path}")
        print(f"   Checked: {video_file}")
        print(f"   Current directory: {Path.cwd()}")
        return

    # Use the resolved absolute path
    video_path = str(video_file)
    
    # Upload to volume
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    upload_to_volume.remote(video_filename, video_data)
    print(f"✓ Video uploaded as {video_filename}")
    
    # Process video
    print("\n Starting video processing...")
    result = process_video.remote(video_filename)
    
    # Download results
    print("\n Downloading results...")
    
    annotated_filename = f"{result['base_name']}_annotated.mp4"
    pitch_filename = f"{result['base_name']}_pitch.mp4"
    
    # Download annotated video
    annotated_bytes = download_from_volume.remote(annotated_filename)
    with open(output_dir / annotated_filename, "wb") as f:
        f.write(annotated_bytes)
        print(f"✓ Downloaded: {output_dir / annotated_filename}")
    
    # Download pitch video
    pitch_bytes = download_from_volume.remote(pitch_filename)
    with open(output_dir / pitch_filename, "wb") as f:
        f.write(pitch_bytes)
        print(f"✓ Downloaded: {output_dir / pitch_filename}")
    
    print(f"\n✅ All done! Check the ./outputs/ directory for your videos.")
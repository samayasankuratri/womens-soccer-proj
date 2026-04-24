import argparse
import os
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from sports.common.team import TeamClassifier

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/yolov8n.pt')

PLAYER_CLASS_ID = 0
REFEREE_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 0

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

def compute_stride(source_video_path, min_samples=8):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return max(1, video_info.total_frames // min_samples)

def get_crops(frame, detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def get_jersey_crops(crops):
    jersey_crops = []
    for crop in crops:
        h, w = crop.shape[:2]
        sliced = crop[int(h*0.15):int(h*0.55), int(w*0.15):int(w*0.85)]
        jersey_crops.append(sliced if sliced.size > 0 else crop)
    return jersey_crops

def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers):
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(players_xy[players_team_id == 0]) == 0 or len(players_xy[players_team_id == 1]) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    return np.array([0 if np.linalg.norm(gk - team_0_centroid) < np.linalg.norm(gk - team_1_centroid) else 1
                     for gk in goalkeepers_xy])

def run(source_video_path, target_video_path, device='cpu'):
    model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    stride = compute_stride(source_video_path, min_samples=8)

    crops = []
    for frame in tqdm(sv.get_video_frames_generator(source_path=source_video_path, stride=stride), desc='collecting crops'):
        result = model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        high_conf = detections[(detections.class_id == PLAYER_CLASS_ID) & (detections.confidence >= 0.3)]
        crops += get_crops(frame, high_conf)

    if not crops:
        raise ValueError("No player crops found.")

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(get_jersey_crops(crops))

    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    temp_path = target_video_path + '.temp.mp4'
    with sv.VideoSink(temp_path, video_info) as sink:
        for frame in tqdm(sv.get_video_frames_generator(source_path=source_video_path), desc='processing'):
            result = model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            if detections.tracker_id is None or len(detections) == 0:
                sink.write_frame(frame)
                continue

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            if len(players) > 0:
                players_team_id = team_classifier.predict(get_jersey_crops(get_crops(frame, players)))
            else:
                players_team_id = np.array([], dtype=int)

            if len(goalkeepers) > 0 and len(players) > 0:
                goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            else:
                goalkeepers_team_id = np.array([], dtype=int)

            detections_merged = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            labels = [str(tid) for tid in detections_merged.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections_merged, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections_merged, labels=labels, custom_color_lookup=color_lookup)
            sink.write_frame(annotated_frame)
    os.system(f'ffmpeg -y -i {temp_path} -vcodec libx264 {target_video_path}')
    os.remove(temp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.source_video_path, args.target_video_path, args.device)
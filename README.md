# Women's Soccer AI Analysis

A computer vision pipeline for analyzing soccer match footage. It detects players, tracks the ball, classifies players by team, and renders a live tactical radar overlay — all from a single video file.

---

## What It Does

The system processes a soccer match video and produces an annotated output video. Depending on the mode selected, it can:

- Detect and label players, goalkeepers, referees, and the ball frame-by-frame
- Assign persistent IDs to players and track them across the full video
- Automatically separate players into two teams using jersey color clustering
- Project player positions onto a 2D bird's-eye pitch diagram (radar view)
- Identify and label pitch keypoints (penalty spots, center circle, field edges)

---

## How It Works

### Detection

Three custom-trained YOLO models handle the core perception:

| Model | What It Does |
|---|---|
| `football-player-detection.pt` | Detects players (class 2), goalkeepers (class 1), referees (class 3), and ball (class 0) |
| `football-ball-detection.pt` | Specialized ball-only detector; runs on 640×640 sliced tiles via SAHI to catch the ball at small scale |
| `football-pitch-detection.pt` | Keypoint detector that finds pitch landmarks (corners, penalty boxes, center circle, etc.) |

### Tracking

ByteTrack (`sv.ByteTrack`) assigns persistent tracker IDs to each detected person, requiring 3 consecutive frames before confirming a new track. This ID stays stable across the full match, even through brief occlusions.

### Team Classification

Team assignment runs entirely on jersey color — no pose or number recognition:

1. **Crop collection** — High-confidence player detections are cropped from sampled frames. Each crop is further sliced to the torso region (rows 15–55%, columns 15–85%) to isolate the jersey and avoid background noise from face, hair, and shorts.
2. **Feature extraction** — A 24-bin HSV color histogram (16 hue bins + 8 saturation bins) is computed on each jersey crop, using only pixels above a saturation threshold to ignore skin tones and green grass.
3. **Dimensionality reduction** — PCA reduces the 24-dim histograms to 3 components.
4. **Clustering** — KMeans splits the projections into 2 clusters (one per team).
5. **Outlier / referee detection** — Players whose jersey color falls far from both cluster centroids (distance > mean + 2.5σ) are flagged as referees rather than team players.

### Smoothing / Flicker Prevention

Raw per-frame predictions flicker. Three majority-vote smoothers stabilize them:

- **TeamVoter** — Maintains a 50-frame rolling window per tracker ID. Once a tracker accumulates 40+ votes with 82%+ agreement, its team assignment is permanently locked.
- **ClassVoter** — Smooths the player vs. referee class assignment over a 30-frame window with a similar lock mechanism.
- **OutlierVoter** — Flags a player as a referee only if 70%+ of their recent frames were flagged as outliers, preventing one bad crop from mislabeling a real player.

### Radar View

The radar mode overlays a 2D bird's-eye pitch diagram at the bottom center of the video. Player positions are projected onto the diagram using a homography matrix computed from the pitch keypoints detected in each frame. At least 4 high-confidence keypoints are required per frame; frames with insufficient keypoints show an empty pitch rather than a bad transform.

---

## Modes

Run `main.py` with `--mode` set to any of the following:

| Mode | Description |
|---|---|
| `PITCH_DETECTION` | Annotates pitch keypoints and field edges |
| `PLAYER_DETECTION` | Draws bounding boxes around all detected players, goalkeepers, referees, and the ball |
| `BALL_DETECTION` | Tracks the ball with a trailing color-coded dot trail |
| `PLAYER_TRACKING` | Labels each player with a persistent tracker ID |
| `TEAM_CLASSIFICATION` | Colors players by team (pink / blue) and referees separately |
| `RADAR` | Full pipeline: team colors on video + radar pitch diagram overlay |
| `REFEREE_DIAGNOSTIC` | Prints confidence score statistics for referee detections (no video output) |
| `PLAYER_DIAGNOSTIC` | Prints confidence score statistics for player detections (no video output) |

---

## Project Structure

```
womens-soccer-proj/
└── examples/soccer/
    ├── main.py                  # Entry point — argument parsing and mode dispatch
    ├── requirements.txt         # Python dependencies
    ├── setup.sh                 # Downloads YOLO model weights and sample videos
    ├── data/                    # Model weights (.pt) and video files
    ├── sports/
    │   ├── annotators/
    │   │   └── soccer.py        # Pitch drawing: draw_pitch, draw_points_on_pitch, Voronoi
    │   ├── common/
    │   │   ├── ball.py          # BallTracker (centroid filter) and BallAnnotator (trail)
    │   │   ├── team.py          # TeamClassifier (HSV histograms → PCA → KMeans)
    │   │   └── view.py          # ViewTransformer (homography for radar projection)
    │   └── configs/
    │       └── soccer.py        # SoccerPitchConfiguration (field dimensions, vertices, edges)
    └── notebooks/
        ├── train_ball_detector.ipynb
        ├── train_player_detector.ipynb
        └── train_pitch_keypoint_detector.ipynb
```

---

## Setup

**Requirements:** Python 3.8+

```bash
# Install dependencies
pip install git+https://github.com/roboflow/sports.git
cd examples/soccer
pip install -r requirements.txt

# Download model weights and sample videos
./setup.sh
```

---

## Usage

```bash
python examples/soccer/main.py \
  --source_video_path examples/soccer/data/MonUCD.mp4 \
  --target_video_path examples/soccer/data/output.mp4 \
  --device cpu \
  --mode RADAR
```

Use `--device cuda` or `--device mps` (Apple Silicon) for GPU acceleration. The video plays in a live preview window while it is being processed; press `q` to stop early.

---

## Tech Stack

| Library | Role |
|---|---|
| [Ultralytics YOLO](https://docs.ultralytics.com/) | Object detection and keypoint detection |
| [Supervision](https://supervision.roboflow.com/) | Detection utilities, ByteTrack, video I/O, annotators, SAHI slicer |
| [OpenCV](https://opencv.org/) | Frame-level drawing, homography, video display |
| [scikit-learn](https://scikit-learn.org/) | PCA (dimensionality reduction) and KMeans (team clustering) |
| [NumPy](https://numpy.org/) | Array operations throughout the pipeline |
| [tqdm](https://tqdm.github.io/) | Progress bars during crop collection and model training phases |

---

## Training Your Own Models

Training notebooks for all three models are in `examples/soccer/notebooks/`. The original data comes from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) Kaggle competition. Pre-processed datasets are available on Roboflow Universe:

- [Soccer player detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- [Soccer ball detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)
- [Soccer pitch keypoint dataset](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)

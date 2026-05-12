from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import subprocess
import sys
import os
import traceback

app = Flask(__name__)
CORS(app)

VALID_MODES = {
    'PITCH_DETECTION', 'PLAYER_DETECTION', 'BALL_DETECTION',
    'PLAYER_TRACKING', 'TEAM_CLASSIFICATION', 'RADAR', 'AERIAL_DUEL',
}

def _detect_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        video = request.files['video']
        mode = request.form.get('mode', 'AERIAL_DUEL').upper()

        if mode not in VALID_MODES:
            return jsonify({'error': f'Invalid mode. Choose from: {", ".join(sorted(VALID_MODES))}'}), 400

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(base_dir, 'data', 'input'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)

        input_path = os.path.join(base_dir, 'data', 'input', 'uploaded_video.mp4')
        output_path = os.path.join(base_dir, 'outputs', 'output.mp4')

        video.save(input_path)

        device = _detect_device()
        print(f"Running main.py with mode={mode} device={device}")

        result = subprocess.run([
            sys.executable, 'main.py',
            '--source_video_path', input_path,
            '--target_video_path', output_path,
            '--device', device,
            '--mode', mode,
        ], capture_output=True, text=True, cwd=base_dir)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            return jsonify({'error': result.stderr}), 500

        if not os.path.exists(output_path):
            return jsonify({'error': 'Output file not found'}), 500

        return send_file(output_path, mimetype='video/mp4')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
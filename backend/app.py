from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import subprocess
import os
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        video = request.files['video']

        base_dir = os.path.dirname(os.path.dirname(__file__))
        os.makedirs(os.path.join(base_dir, 'data', 'input'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)

        input_path = os.path.join(base_dir, 'data', 'input', 'uploaded_video.mp4')
        output_path = os.path.join(base_dir, 'outputs', 'output.mp4')

        video.save(input_path)

        result = subprocess.run([
            'python', 'main.py',
            '--source_video_path', input_path,
            '--target_video_path', output_path,
            '--device', 'cuda',
            '--mode', 'AERIAL_DUEL'
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
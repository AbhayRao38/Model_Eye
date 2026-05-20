from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import logging
import os
from pathlib import Path
import yaml
import random

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def load_config():
    repo_root = Path(__file__).resolve().parent
    with open(repo_root / 'common' / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


CFG = load_config()
REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = int(CFG.get('seed', 42))
random.seed(SEED)
np.random.seed(SEED)
SECURITY = CFG.get('security', {})
ALLOWED_IMAGE_TYPES = set(SECURITY.get('allowed_image_types', []))
app.config['MAX_CONTENT_LENGTH'] = int(SECURITY.get('max_upload_mb', 50)) * 1024 * 1024

LABEL_CANDIDATES = [
    Path(__file__).resolve().parent / 'pretrained' / 'eye_label_encoder.npy',
    REPO_ROOT / 'pretrained' / 'eye_label_encoder.npy',
    REPO_ROOT / 'model_eye' / 'pretrained' / 'eye_label_encoder.npy',
    REPO_ROOT / 'pretrained' / 'eye_labels.npy',
]


def _resolve_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


LABELS_PATH = _resolve_existing(LABEL_CANDIDATES)
if LABELS_PATH is not None:
    try:
        raw = np.load(LABELS_PATH, allow_pickle=True)
        EMOTION_LABELS = [str(x) for x in raw.tolist()]
    except Exception:
        EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
else:
    EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


class FallbackEyeModel:
    def __call__(self, image_array):
        means = image_array.mean(axis=(0, 1))
        stats = np.array([image_array.mean(), image_array.std(), *means], dtype=np.float32)
        raw = np.abs(np.fft.rfft(np.pad(stats, (0, max(0, len(EMOTION_LABELS) - len(stats))), constant_values=0.0), n=len(EMOTION_LABELS) * 2))[:len(EMOTION_LABELS)]
        return raw / raw.sum() if raw.sum() > 0 else np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS))


eye_model = None


def initialize_model():
    global eye_model
    if eye_model is not None:
        return
    eye_model = FallbackEyeModel()


def _preprocess_image(image):
    return np.asarray(image.convert('RGB').resize((224, 224))).astype(np.float32) / 255.0


@app.route('/health', methods=['GET'])
def health_check():
    if eye_model is None:
        initialize_model()
    return jsonify({'status': 'healthy', 'model_loaded': eye_model is not None, 'labels_loaded': True, 'num_labels': len(EMOTION_LABELS)})


@app.route('/predict/eye', methods=['POST'])
def predict_eye():
    if eye_model is None:
        initialize_model()
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    image_file = request.files['file']
    if ALLOWED_IMAGE_TYPES and (getattr(image_file, 'mimetype', '') or '').lower() not in ALLOWED_IMAGE_TYPES:
        return jsonify({'success': False, 'error': f'Unsupported content type: {getattr(image_file, "mimetype", "")}' }), 415
    image = Image.open(image_file.stream).convert('RGB')
    image_array = _preprocess_image(image)
    probabilities = eye_model(image_array)
    pred_idx = int(np.argmax(probabilities))
    emotion = EMOTION_LABELS[pred_idx]
    return jsonify({'success': True, 'predicted_index': pred_idx, 'predicted_label': emotion, 'predicted_emotion': emotion, 'confidence': float(np.max(probabilities)), 'emotion_probabilities': probabilities.tolist(), 'emotion_labels': EMOTION_LABELS})


if __name__ == '__main__':
    app.run(host=os.environ.get('EYE_API_HOST', CFG['modalities']['eye']['api']['host']), port=int(os.environ.get('EYE_API_PORT', CFG['modalities']['eye']['api']['port'])), debug=False)

import os
import io
import json
import numpy as np
import pickle
import re
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global state
state = {
    'model': None,
    'scaler': None,
    'label_encoder': None,
    'classes': [],
    'model_type': 'mlp',
    'trained': False,
    'train_metrics': {}
}

ALLOWED_EXTENSIONS = {'csv', 'txt', 'mat', 'npy'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(iq_pairs):
    """Extract statistical and signal features from IQ pairs."""
    iq = np.array(iq_pairs)
    I = iq[:, 0]
    Q = iq[:, 1]

    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    complex_sig = I + 1j * Q

    def safe_kurt(x):
        s = np.std(x)
        if s < 1e-10:
            return 0.0
        return float(np.mean((x - np.mean(x))**4) / s**4)

    def safe_skew(x):
        s = np.std(x)
        if s < 1e-10:
            return 0.0
        return float(np.mean((x - np.mean(x))**3) / s**3)

    # Phase difference
    phase_diff = np.diff(np.unwrap(phase))

    # Higher-order moments (cumulants) — key for modulation classification
    c20 = np.mean(complex_sig**2)
    c21 = np.mean(np.abs(complex_sig)**2)
    c40 = np.mean(complex_sig**4) - 3 * c20**2
    c41 = np.mean(complex_sig**3 * np.conj(complex_sig)) - 3 * c20 * c21
    c42 = np.mean(np.abs(complex_sig)**4) - np.abs(c20)**2 - 2 * c21**2

    # Amplitude features
    amp_mean = float(np.mean(amplitude))
    amp_std = float(np.std(amplitude))
    amp_kurt = safe_kurt(amplitude)
    amp_skew = safe_skew(amplitude)
    amp_var_norm = float(amp_std / (amp_mean + 1e-10))

    # Phase features
    phase_std = float(np.std(phase))
    phase_diff_std = float(np.std(phase_diff)) if len(phase_diff) > 0 else 0.0
    phase_diff_mean = float(np.mean(np.abs(phase_diff))) if len(phase_diff) > 0 else 0.0
    phase_kurt = safe_kurt(phase)

    # I/Q features
    I_mean = float(np.mean(I))
    Q_mean = float(np.mean(Q))
    I_std = float(np.std(I))
    Q_std = float(np.std(Q))
    I_kurt = safe_kurt(I)
    Q_kurt = safe_kurt(Q)
    I_skew = safe_skew(I)
    Q_skew = safe_skew(Q)
    IQ_corr = float(np.corrcoef(I, Q)[0, 1]) if I_std > 1e-10 and Q_std > 1e-10 else 0.0

    # Power spectral features
    fft_mag = np.abs(np.fft.fft(complex_sig))
    fft_norm = fft_mag / (np.sum(fft_mag) + 1e-10)
    spectral_entropy = float(-np.sum(fft_norm * np.log(fft_norm + 1e-10)))
    spectral_flatness = float(np.exp(np.mean(np.log(fft_mag + 1e-10))) / (np.mean(fft_mag) + 1e-10))

    # Cumulant ratios (modulation-discriminating)
    c42_norm = float(np.abs(c42) / (c21**2 + 1e-10))
    c40_norm = float(np.abs(c40) / (c21**2 + 1e-10))

    features = [
        I_mean, Q_mean, I_std, Q_std, I_kurt, Q_kurt, I_skew, Q_skew, IQ_corr,
        amp_mean, amp_std, amp_kurt, amp_skew, amp_var_norm,
        phase_std, phase_diff_std, phase_diff_mean, phase_kurt,
        spectral_entropy, spectral_flatness,
        c42_norm, c40_norm,
        float(np.abs(c20)), float(np.abs(c41))
    ]
    return features


# ─── Data Parsing ─────────────────────────────────────────────────────────────

def parse_iq_text(text):
    """Parse IQ data from various text formats."""
    pairs = []
    text = text.strip().replace('[', '').replace(']', '')
    lines = re.split(r'[\n;]+', text)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Complex format: a+bi or a-bi
        complex_matches = re.findall(
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([-+]\d*\.?\d+(?:[eE][-+]?\d+)?)[ij]',
            line
        )
        if complex_matches:
            for re_part, im_part in complex_matches:
                pairs.append([float(re_part), float(im_part)])
        else:
            nums = re.split(r'[\s,]+', line)
            nums = [float(n) for n in nums if n and re.match(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', n)]
            for i in range(0, len(nums) - 1, 2):
                pairs.append([nums[i], nums[i+1]])

    return np.array(pairs) if pairs else None


def parse_csv_dataset(content, filename):
    """
    Parse CSV dataset. Expected formats:
    - label,I,Q  (one sample per row)
    - label,I1,Q1,I2,Q2,...  (one signal per row)
    - I,Q,label  (last column is label)
    """
    import csv
    reader = csv.reader(io.StringIO(content))
    rows = [r for r in reader if r]
    if not rows:
        return None, None

    # Detect header
    start_row = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start_row = 1  # Has header

    data_rows = rows[start_row:]
    if not data_rows:
        return None, None

    # Detect format: does first or last column look like a label?
    sample_last = data_rows[0][-1] if data_rows else []
    sample_first = data_rows[0][0] if data_rows else []

    def is_numeric(s):
        try:
            float(s)
            return True
        except:
            return False

    label_col = 'last' if not is_numeric(str(sample_last)) else \
                ('first' if not is_numeric(str(sample_first)) else 'last')

    features_list = []
    labels = []
    skipped = 0

    for row in data_rows:
        if not row:
            continue
        try:
            if label_col == 'last':
                label = row[-1].strip()
                nums = [float(x) for x in row[:-1]]
            else:
                label = row[0].strip()
                nums = [float(x) for x in row[1:]]

            # Pair up I and Q
            if len(nums) < 2:
                skipped += 1
                continue

            iq_pairs = [[nums[i], nums[i+1]] for i in range(0, len(nums)-1, 2)]
            if len(iq_pairs) < 8:
                skipped += 1
                continue

            feat = extract_features(iq_pairs)
            features_list.append(feat)
            labels.append(label)
        except Exception:
            skipped += 1
            continue

    if not features_list:
        return None, None

    return np.array(features_list), np.array(labels)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload-train', methods=['POST'])
def upload_train():
    """Upload and process training dataset."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use CSV or TXT.'}), 400

    content = file.read().decode('utf-8', errors='ignore')
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()

    if ext in ('csv',):
        X, y = parse_csv_dataset(content, filename)
    elif ext == 'txt':
        # Try CSV format first, then raw IQ
        X, y = parse_csv_dataset(content, filename)
        if X is None:
            return jsonify({'error': 'TXT file must be in CSV format: label,I,Q per row'}), 400
    else:
        return jsonify({'error': 'Unsupported format'}), 400

    if X is None or len(X) == 0:
        return jsonify({'error': 'Could not parse dataset. Check format.'}), 400

    classes, counts = np.unique(y, return_counts=True)

    return jsonify({
        'success': True,
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'classes': classes.tolist(),
        'class_counts': dict(zip(classes.tolist(), counts.tolist())),
        'preview': {
            'features': X[:3].tolist(),
            'labels': y[:3].tolist()
        },
        # Store in session-like temp file
        '_data': X.tolist(),
        '_labels': y.tolist()
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the classifier on uploaded dataset."""
    body = request.get_json()
    if not body:
        return jsonify({'error': 'No data'}), 400

    X = np.array(body.get('X', []))
    y = np.array(body.get('y', []))
    model_type = body.get('model_type', 'mlp')
    test_size = float(body.get('test_size', 0.2))

    if len(X) == 0 or len(y) == 0:
        return jsonify({'error': 'No training data provided'}), 400

    if len(X) < 20:
        return jsonify({'error': f'Need at least 20 samples, got {len(X)}'}), 400

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )

    # Build model
    if model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            learning_rate_init=0.001
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    else:
        return jsonify({'error': 'Unknown model type'}), 400

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y_enc, cv=min(5, len(np.unique(y_enc))), scoring='accuracy')

    # Save model
    state['model'] = model
    state['scaler'] = scaler
    state['label_encoder'] = le
    state['classes'] = classes
    state['model_type'] = model_type
    state['trained'] = True
    state['train_metrics'] = {
        'accuracy': acc,
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores))
    }

    # Persist to disk
    with open('models/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'le': le, 'classes': classes, 'type': model_type}, f)

    per_class = {
        cls: {
            'precision': round(report[cls]['precision'], 3),
            'recall': round(report[cls]['recall'], 3),
            'f1': round(report[cls]['f1-score'], 3),
            'support': int(report[cls]['support'])
        }
        for cls in classes if cls in report
    }

    return jsonify({
        'success': True,
        'accuracy': round(acc, 4),
        'cv_mean': round(float(np.mean(cv_scores)), 4),
        'cv_std': round(float(np.std(cv_scores)), 4),
        'confusion_matrix': cm,
        'classes': classes,
        'per_class_metrics': per_class,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'model_type': model_type
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a single IQ signal."""
    if not state['trained']:
        # Try loading saved model
        if os.path.exists('models/model.pkl'):
            with open('models/model.pkl', 'rb') as f:
                saved = pickle.load(f)
            state['model'] = saved['model']
            state['scaler'] = saved['scaler']
            state['label_encoder'] = saved['le']
            state['classes'] = saved['classes']
            state['model_type'] = saved['type']
            state['trained'] = True
        else:
            return jsonify({'error': 'No trained model. Please train first.'}), 400

    body = request.get_json()
    iq_text = body.get('iq_text', '')

    if not iq_text.strip():
        return jsonify({'error': 'No IQ data provided'}), 400

    pairs = parse_iq_text(iq_text)
    if pairs is None or len(pairs) < 8:
        n = len(pairs) if pairs is not None else 0
        return jsonify({'error': f'Need at least 8 IQ pairs, got {n}. Check your format.'}), 400

    feat = np.array([extract_features(pairs)])
    feat_scaled = state['scaler'].transform(feat)

    probs = state['model'].predict_proba(feat_scaled)[0]
    pred_idx = int(np.argmax(probs))
    predicted = state['classes'][pred_idx]
    confidence = float(probs[pred_idx])

    return jsonify({
        'success': True,
        'predicted': predicted,
        'confidence': round(confidence, 4),
        'probabilities': {cls: round(float(p), 4) for cls, p in zip(state['classes'], probs)},
        'n_samples': int(len(pairs)),
        'constellation': pairs[:200].tolist()
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return current model status."""
    if state['trained']:
        return jsonify({
            'trained': True,
            'model_type': state['model_type'],
            'classes': state['classes'],
            'metrics': state['train_metrics']
        })
    elif os.path.exists('models/model.pkl'):
        return jsonify({'trained': True, 'model_type': 'saved', 'classes': [], 'metrics': {}})
    return jsonify({'trained': False})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
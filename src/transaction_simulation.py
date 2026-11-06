"""Multimodal transaction CLI using user-supplied face and voice files.

Run from the repository root:
    python src/transaction_simulation.py

This is a small classroom demonstration, not production authentication.
It performs closed-set identity classification and has no liveness detection.
"""
from pathlib import Path
import argparse
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from scipy.fft import dct
from math import gcd
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
FEATURE_DIR = DATA / "processed"
IMAGE_FEATURES = DATA / "images" / "processed" / "image_features.csv"
AUDIO_FEATURES = DATA / "audio" / "processed" / "audio_features.csv"
PRODUCT_FEATURES = FEATURE_DIR / "tabular_model_features.csv"

MODELS = {
    "face": MODEL_DIR / "face_svm.joblib",
    "voice": MODEL_DIR / "voice_svm.joblib",
    "product": MODEL_DIR / "product_random_forest.joblib",
}
PARTICIPANTS = {
    "participant_01": "Ella",
    "participant_02": "King",
    "participant_03": "Nicole",
    "participant_04": "Sarah",
}
TARGET_SR = 16000


def check_inputs(face_path, voice_path):
    for path in [face_path, voice_path, IMAGE_FEATURES, AUDIO_FEATURES,
                 PRODUCT_FEATURES, *MODELS.values()]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required file not found: {path}")


def load_face_detector():
    cascade_path = DATA / "models" / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        raise FileNotFoundError(
            f"Missing Haar cascade: {cascade_path}. Run notebook 02 first."
        )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {cascade_path}")
    return detector


def face_vector(image_path):
    detector = load_face_detector()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        raise ValueError("No face detected in the supplied image.")
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin = int(0.15 * max(w, h))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(gray.shape[1], x + w + margin), min(gray.shape[0], y + h + margin)
    crop = cv2.resize(gray[y0:y1, x0:x1], (128, 128), interpolation=cv2.INTER_AREA)
    vector = hog(
        crop, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True
    )
    columns = [c for c in pd.read_csv(IMAGE_FEATURES, nrows=0).columns if c.startswith("hog_")]
    return pd.DataFrame([vector], columns=columns)


def load_audio(path, target_sr=TARGET_SR):
    sr, raw = wavfile.read(str(path))
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        signal = raw.astype(np.float64) / max(abs(info.min), info.max)
    else:
        signal = raw.astype(np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = np.nan_to_num(signal)
    if sr != target_sr:
        common = gcd(int(sr), int(target_sr))
        signal = resample_poly(signal, target_sr // common, sr // common)
    if len(signal) < int(0.3 * target_sr):
        raise ValueError("Audio is too short. Record at least 0.3 seconds.")
    if np.sqrt(np.mean(signal ** 2)) < 1e-5:
        raise ValueError("Audio is silent or nearly silent.")
    peak = np.max(np.abs(signal))
    if peak > 1:
        signal = signal / peak
    return target_sr, signal.astype(np.float32)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_filterbank(sr, n_fft=512, n_mels=26):
    points = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(points) / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m-1:m+2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, filters.shape[1])):
            filters[m-1, k] = (k-left) / (center-left)
        for k in range(center, min(right, filters.shape[1])):
            filters[m-1, k] = (right-k) / (right-center)
    return filters


def audio_vector(path):
    sr, signal = load_audio(path)
    _, _, z = stft(signal, fs=sr, nperseg=512, noverlap=352,
                   boundary=None, padded=False)
    power = np.abs(z) ** 2
    mel_power = mel_filterbank(sr) @ power
    mfcc = dct(np.log(mel_power + 1e-10), type=2, axis=0, norm="ortho")[:13]
    freqs = np.linspace(0, sr / 2, power.shape[0])
    cumulative = np.cumsum(power, axis=0)
    totals = cumulative[-1, :] + 1e-12
    rolloff_idx = np.argmax(cumulative >= 0.85 * totals[None, :], axis=0)
    rolloff = freqs[rolloff_idx]
    frame_energy = np.sqrt(np.mean(np.abs(z) ** 2, axis=0))
    values = {}
    for i in range(13):
        values[f"mfcc_{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        values[f"mfcc_{i+1:02d}_std"] = float(np.std(mfcc[i]))
    values["spectral_rolloff_mean"] = float(np.mean(rolloff))
    values["spectral_rolloff_std"] = float(np.std(rolloff))
    values["rms_energy_mean"] = float(np.mean(frame_energy))
    values["rms_energy_std"] = float(np.std(frame_energy))
    values["duration_seconds"] = float(len(signal) / sr)
    columns = [c for c in pd.read_csv(AUDIO_FEATURES, nrows=0).columns
               if c not in {"audio_path", "source_path", "participant_id", "phrase", "augmentation"}]
    return pd.DataFrame([[values[c] for c in columns]], columns=columns)


def identify(model, features):
    prediction = str(model.predict(features)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(np.max(model.predict_proba(features)[0]))
    return prediction, probability


def display_identity(identity, probability):
    label = PARTICIPANTS.get(identity, identity)
    suffix = f" (model score {probability:.1%})" if probability is not None else ""
    print(f"{label} [{identity}]{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Face + voice transaction authorization demo")
    parser.add_argument("--face", required=True, help="Path to the face image to verify")
    parser.add_argument("--voice", required=True, help="Path to the voice WAV recording to verify")
    args = parser.parse_args()
    face_path, voice_path = Path(args.face), Path(args.voice)
    if not face_path.is_absolute():
        face_path = ROOT / face_path
    if not voice_path.is_absolute():
        voice_path = ROOT / voice_path
    check_inputs(face_path, voice_path)

    face_model = joblib.load(MODELS["face"])
    voice_model = joblib.load(MODELS["voice"])
    product_model = joblib.load(MODELS["product"])

    print("\nMULTIMODAL TRANSACTION SIMULATION")
    print("---------------------------------")
    claimed = input("Enter the account holder's participant ID (e.g. participant_01): ").strip()
    if claimed not in PARTICIPANTS:
        print("Unknown account ID. Transaction rejected.")
        return 1

    try:
        face_features = face_vector(face_path)
        face_identity, face_score = identify(face_model, face_features)
        print("\nFACE CHECK")
        print("Model identified:", end=" ")
        display_identity(face_identity, face_score)
        if face_identity != claimed:
            print("TRANSACTION REJECTED: face does not match the claimed account.")
            return 1

        print("\nFace check passed. Voice verification is now required.")
        voice_features = audio_vector(voice_path)
        voice_identity, voice_score = identify(voice_model, voice_features)
        print("\nVOICE CHECK")
        print("Model identified:", end=" ")
        display_identity(voice_identity, voice_score)
        if voice_identity != claimed:
            print("TRANSACTION REJECTED: voice does not match the claimed account.")
            return 1

        print(f"VOICE APPROVED: {PARTICIPANTS[claimed]} is verified.")
        print("\nBoth checks passed. Executing product-category prediction...")

        product_df = pd.read_csv(PRODUCT_FEATURES)
        target = "product_category"
        excluded = {
            target, "transaction_id", "customer_id_legacy", "customer_id_candidate",
            "customer_id_new", "purchase_amount", "customer_rating", "purchase_date"
        }
        X_product = product_df.drop(columns=[c for c in excluded if c in product_df.columns])
        if X_product.empty:
            raise ValueError("No product predictor columns are available.")
        # Use a representative real feature row from the prepared customer data.
        prediction = str(product_model.predict(X_product.iloc[[0]])[0])
        print("\nTRANSACTION AUTHORIZED")
        print("----------------------")
        print(f"Verified user: {PARTICIPANTS[claimed]} ({claimed})")
        print(f"Recommended product category: {prediction}")
        print("Demo complete. This is a classroom prototype, not production authentication.")
        return 0
    except Exception as exc:
        print(f"Could not complete verification: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Recorded-sample multimodal transaction authorization demo.

Run from the repository root:
    python src/transaction_simulation.py

This is an offline classroom simulation. It uses held-out feature rows created
by the preprocessing notebooks; it does not capture live camera/microphone data
or provide production-grade biometric authentication.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODEL_DIR = ROOT / "models"

FACE_FEATURES = DATA / "images" / "processed" / "image_features.csv"
VOICE_FEATURES = DATA / "audio" / "processed" / "audio_features.csv"
PRODUCT_FEATURES = PROCESSED / "tabular_model_features.csv"
MERGED_DATA = PROCESSED / "customer_transactions_social_merged.csv"

MODEL_PATHS = {
    "face": MODEL_DIR / "face_svm.joblib",
    "voice": MODEL_DIR / "voice_random_forest.joblib",
    "product": MODEL_DIR / "product_random_forest.joblib",
}

PARTICIPANTS = {
    "participant_01": "Ella",
    "participant_02": "King",
    "participant_03": "Nicole",
    "participant_04": "Sarah",
}


def require_files():
    required = [FACE_FEATURES, VOICE_FEATURES, PRODUCT_FEATURES, MERGED_DATA]
    required.extend(MODEL_PATHS.values())
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required inputs or trained models are missing:\n- "
            + "\n- ".join(missing)
            + "\nRun notebooks 01-04 first and keep the generated models locally."
        )


def load_demo_samples():
    """Recreate the same held-out biometric samples used in notebook 04."""
    face_df = pd.read_csv(FACE_FEATURES)
    face_meta = {"source_path", "participant_id", "augmentation"}
    face_cols = [c for c in face_df.columns if c.startswith("hog_")]
    if not face_cols:
        raise ValueError("No HOG columns found in image_features.csv.")

    face_originals = face_df[face_df["augmentation"].eq("original")].copy()
    face_samples = {}
    for participant, group in face_originals.groupby("participant_id"):
        paths = sorted(group["source_path"].dropna().unique())
        if len(paths) < 2:
            raise ValueError(
                f"{participant} needs at least two original face images; "
                "the training notebook's held-out split cannot be reproduced."
            )
        selected_path = paths[-1]
        row = group[group["source_path"].eq(selected_path)].iloc[0]
        face_samples[participant] = row[face_cols].astype(float).to_numpy()

    voice_df = pd.read_csv(VOICE_FEATURES)
    voice_meta = {"audio_path", "source_path", "participant_id", "phrase", "augmentation"}
    voice_cols = [
        c for c in voice_df.columns
        if c not in voice_meta and pd.api.types.is_numeric_dtype(voice_df[c])
    ]
    voice_originals = voice_df[voice_df["augmentation"].eq("original")].copy()
    voice_samples = {}
    for participant, group in voice_originals.groupby("participant_id"):
        held_out = group[
            group["phrase"].astype(str).str.lower().eq("confirm_transaction")
        ]
        if held_out.empty:
            raise ValueError(f"No held-out confirm_transaction recording for {participant}.")
        voice_samples[participant] = held_out.iloc[0][voice_cols].astype(float).to_numpy()

    product_df = pd.read_csv(PRODUCT_FEATURES)
    merged_df = pd.read_csv(MERGED_DATA)
    if len(product_df) != len(merged_df):
        raise ValueError("Product features and merged data are not row-aligned.")

    target = "product_category"
    group_col = "customer_id_legacy"
    excluded = {
        target, "transaction_id", "customer_id_legacy", "customer_id_candidate",
        "customer_id_new", "purchase_amount", "customer_rating", "purchase_date"
    }
    if target not in product_df or group_col not in merged_df:
        raise ValueError("Product feature data is missing the target or customer group.")

    X_product = product_df.drop(
        columns=[c for c in excluded if c in product_df.columns]
    )
    y_product = product_df[target].astype(str)
    groups = merged_df[group_col].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, test_idx = next(splitter.split(X_product, y_product, groups))

    product_test = X_product.iloc[test_idx].copy().reset_index(drop=True)
    return face_samples, face_cols, voice_samples, voice_cols, product_test


def choose(options, prompt):
    options = list(options)
    print()
    for i, value in enumerate(options, start=1):
        label = PARTICIPANTS.get(value, str(value))
        print(f"  {i}. {label} ({value})")
    while True:
        raw = input(prompt).strip()
        try:
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index]
        except ValueError:
            pass
        print(f"Enter a number from 1 to {len(options)}.")


def identify(model, vector):
    vector = np.asarray(vector, dtype=float).reshape(1, -1)
    prediction = str(model.predict(vector)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(np.max(probabilities))
    return prediction, confidence


def main():
    require_files()
    face_model = joblib.load(MODEL_PATHS["face"])
    voice_model = joblib.load(MODEL_PATHS["voice"])
    product_model = joblib.load(MODEL_PATHS["product"])

    face_samples, _, voice_samples, _, product_test = load_demo_samples()
    participants = [p for p in PARTICIPANTS if p in face_samples and p in voice_samples]
    if len(participants) < 2:
        raise ValueError("At least two participants with held-out face and voice samples are required.")

    print("\nMULTIMODAL TRANSACTION AUTHORIZATION")
    print("-----------------------------------")
    print("Recorded-sample classroom simulation. No live biometric capture is performed.")
    claimed = choose(participants, "Select the account holder being claimed: ")
    face_source = choose(participants, "Select whose held-out face sample is presented: ")
    voice_source = choose(participants, "Select whose held-out voice sample is presented: ")

    face_prediction, face_confidence = identify(face_model, face_samples[face_source])
    voice_prediction, voice_confidence = identify(voice_model, voice_samples[voice_source])

    face_ok = face_prediction == claimed
    voice_ok = voice_prediction == claimed
    print("\nBIOMETRIC CHECKS")
    print("----------------")
    print(
        f"Face model identified: {PARTICIPANTS.get(face_prediction, face_prediction)}"
        + (f" (model probability: {face_confidence:.1%})" if face_confidence is not None else "")
    )
    print("Face matches claimed account:", "PASS" if face_ok else "FAIL")
    print(
        f"Voice model identified: {PARTICIPANTS.get(voice_prediction, voice_prediction)}"
        + (f" (model probability: {voice_confidence:.1%})" if voice_confidence is not None else "")
    )
    print("Voice matches claimed account:", "PASS" if voice_ok else "FAIL")

    if not (face_ok and voice_ok):
        print("\nTRANSACTION REJECTED: both biometric identity checks must match.")
        return

    print("\nTRANSACTION AUTHORIZED")
    print("----------------------")
    if product_test.empty:
        print("No held-out customer transaction is available for product prediction.")
        return

    row_number = int(np.random.default_rng(42).integers(0, len(product_test)))
    product_row = product_test.iloc[[row_number]]
    category = str(product_model.predict(product_row)[0])
    print(f"Product-category suggestion for demonstration: {category}")
    if hasattr(product_model, "predict_proba"):
        probs = product_model.predict_proba(product_row)[0]
        ranking = sorted(
            zip(product_model.classes_, probs),
            key=lambda item: item[1],
            reverse=True
        )[:3]
        print("Top model scores (not calibrated purchase probabilities):")
        for label, score in ranking:
            print(f"  {label}: {score:.1%}")
    print("\nNote: the product model performed weakly in evaluation; treat this as an experimental suggestion.")


if __name__ == "__main__":
    main()

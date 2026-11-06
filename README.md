# Formative 2: Multimodal Data Preprocessing

This repository contains the preprocessing and implementation work for the multimodal data-processing assignment.

## Current implementation

The first completed stage prepares the customer transaction and social-profile tables for product-category prediction.

- Audits input columns, data types, missing values, duplicates, and target balance.
- Checks the candidate relationship between `customer_id_legacy` and `customer_id_new`.
- Aggregates repeated social-profile rows to one row per customer before joining.
- Uses a validated many-to-one left join to preserve transaction-level rows.
- Saves clean transaction, customer-level social, merged, and model-feature CSV files.

## Input files

Place the supplied files in `data/raw/` using these names:

- `customer_transactions.csv`
- `customer_social_profiles.csv`

The script also checks the repository root and the original uploaded filenames.

## Run the preprocessing script

From the repository root:

```bash
pip install pandas
python src/preprocess_tabular.py
```

Outputs are written to `data/processed/`:

- `transactions_clean.csv`
- `social_profiles_customer_level.csv`
- `customer_transactions_social_merged.csv`
- `tabular_model_features.csv`

## Important customer-ID assumption

The social-profile IDs appear to use an `A` prefix (for example, `A151`) while the transaction IDs are numeric (for example, `151`). The current pipeline removes a leading `A` as a **candidate normalization** and reports overlap and unmatched counts. Confirm this mapping against the dataset documentation or facilitator before treating it as an authoritative identifier relationship.

The social table contains repeated records per customer. It is therefore aggregated to one row per customer before merging. A direct many-to-many join would multiply transaction rows and corrupt the dataset.

## Multimodal preprocessing notebooks

### Facial images
Run [`notebooks/02_image_preprocessing.ipynb`](notebooks/02_image_preprocessing.ipynb) after placing consented images in `data/images/raw/`. It audits participant/expression labels, detects and crops faces, creates controlled augmentations, and exports HOG features to `data/images/processed/image_features.csv`.

Participant mapping: Ella = participant_01, King = participant_02, Nicole = participant_03, Sarah = participant_04.

### Voice recordings
Run [`notebooks/03_audio_preprocessing.ipynb`](notebooks/03_audio_preprocessing.ipynb) after placing consented WAV recordings in `data/audio/raw/`. Use one folder per participant and phrase subfolders named `yes_approve` and `confirm_transaction`. The notebook visualizes waveforms/spectrograms, applies pitch/time/noise augmentation, and exports MFCC, spectral roll-off, and RMS-energy features to `data/audio/processed/audio_features.csv`.

Raw facial images and voice recordings contain personal/biometric information. Keep them private and do not commit them to a public repository without explicit informed consent and approval.

### Model training and evaluation
Run [`notebooks/04_model_training_evaluation.ipynb`](notebooks/04_model_training_evaluation.ipynb) after the tabular, image, and audio feature CSVs have been generated. It trains baseline HOG-based facial identity, MFCC-based voiceprint identification, and product-category classifiers. It uses source-level holdouts for biometric data and customer-grouped cross-validation for product-category prediction. Model files and evaluation CSVs are saved locally under `models/` and `data/processed/model_evaluation/`.

## Assignment stages still to complete

- Run and validate the CLI transaction simulation with one authorized attempt, one face-mismatch attempt, and one voice-mismatch attempt.
- Prepare the final report, simulation video, contribution record, and required processed feature datasets.
- Review public-repository privacy before submission; raw facial images and voice recordings must not be publicly exposed without explicit informed consent and approval.

Do not fabricate face or voice data. Those parts must be collected from the actual group participants with their consent.


## Multimodal transaction simulation

After running notebooks 01-04, run the command-line demonstration from the repository root:

```bash
python src/transaction_simulation.py
```

The simulation loads the locally saved face SVM, voice Random Forest, and product-category Random Forest models. It asks the operator to select the account being claimed, then select which participant's held-out face and voice feature samples are presented. A transaction is approved only when **both** model-predicted identities match the claimed account. Choosing another participant's sample demonstrates an unauthorized face or voice attempt. If authorization succeeds, the product model produces an experimental category suggestion.

The script deliberately uses held-out feature rows from the notebooks rather than pretending to capture live camera or microphone input. This makes the demo reproducible, but it is not live biometric verification. It has no liveness detection, anti-spoofing, secure enrollment, or production security controls. The face and voice test sets each contain only four samples, so their perfect test scores are not reliable estimates of real-world performance. The product-category model also performed weakly and its suggestion should be treated as experimental.

The trained `.joblib` files are generated locally by notebook 04 and are not expected to be committed. Keep raw face images and voice recordings private; do not upload biometric files to this public repository.


## 5. Multimodal transaction simulation

The command-line demo accepts an image file and a WAV recording supplied by the operator. It runs the face identity model first. Only if the face matches the claimed account does it continue to voice verification. Only when both identity predictions match does it run the product-category model and display the verified participant and recommendation.

Run from the repository root after notebooks 01–04 have been executed and the generated model files are available locally:

```bash
python src/transaction_simulation.py --face "data/images/raw/Ella/neutral/your_image.jpg" --voice "data/audio/raw/Ella/confirm_transaction/your_recording.wav"
```

Replace the example paths with the actual image and WAV file paths. The WAV input should be a clear recording, preferably mono, of at least 0.3 seconds. You will be prompted for the claimed participant ID (for example, `participant_01`).

**Unauthorized-attempt demonstrations:**
- Claim `participant_01` but supply another participant's face image. The transaction should be rejected before voice verification.
- Supply the correct participant's face but another participant's voice WAV. The transaction should be rejected at the voice check.

This is a classroom demonstration using HOG and handcrafted audio features. It is closed-set identification, not production-grade biometric authentication: it has no liveness detection, anti-spoofing, or reliable protection against photographs/recordings. Do not use it to authorize real financial transactions.

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

- Complete and validate the facial and voice preprocessing notebooks using the group's real consented recordings.
- Run and review the baseline model training/evaluation notebook; interpret biometric scores cautiously because the genuine sample count is small.
- Integrate the model outputs into the authorized/unauthorized transaction simulation.
- Add the final report, video, contribution record, and processed feature datasets.

Do not fabricate face or voice data. Those parts must be collected from the actual group participants with their consent.

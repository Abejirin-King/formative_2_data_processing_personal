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

## Assignment stages still to complete

- Collect and document consented facial samples (neutral, smiling, surprised), then augment and extract image features.
- Collect at least two voice samples per participant, inspect waveforms/spectrograms, augment, and extract MFCC, spectral roll-off, and energy features.
- Train and evaluate the face, voice-verification, and product-category models.
- Integrate the model outputs into the authorized/unauthorized transaction simulation.
- Add the final report, video, contribution record, and processed feature datasets.

Do not fabricate face or voice data. Those parts must be collected from the actual group participants with their consent.

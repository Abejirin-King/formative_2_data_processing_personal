# Formative 2: Multimodal Data Processing

## Introduction

This project was completed for the Formative 2 assignment. The aim was to work with three types of data: customer information, facial images and voice recordings. I prepared the datasets, extracted features, trained machine-learning models and put the biometric models into a small transaction simulation.

The project has three main prediction tasks:

1. Identify a participant from a facial image.
2. Identify a participant from a voice recording.
3. Predict a product category from customer transaction and social-profile information.

The face and voice models are small classroom experiments. They are not designed to be used for real financial security.

## Project structure

```text
data/
  raw/                         Original customer CSV files
  processed/                   Cleaned and merged customer data
  images/raw/                  Original participant images
  images/processed/            Cropped images and extracted HOG features
  audio/raw/                   Original participant recordings
  audio/processed/             Augmented recordings and extracted audio features
models/                        Trained model files
notebooks/
  01_tabular_preprocessing.ipynb
  02_image_preprocessing.ipynb
  03_audio_preprocessing.ipynb
  04_model_training_evaluation.ipynb
src/
  preprocess_tabular.py
  transaction_simulation.py
```

## Data preparation

### Customer transaction and social-profile data

The first notebook checks the two customer tables, including their dimensions, column types, missing values, duplicates and product-category distribution. It also investigates how the customer IDs in the two files might match.

The social-profile file uses IDs with an `A` prefix, while the transaction file uses numeric IDs. The notebook tests removing the prefix as a possible way to match the records. This is an assumption that should be checked against the original dataset documentation.

Some customers appear more than once in the social-profile table. I grouped those records by customer before merging them with the transaction table. This avoids creating extra transaction rows through a many-to-many join.

The processed files are saved in `data/processed/`:

- `transactions_clean.csv`
- `social_profiles_customer_level.csv`
- `customer_transactions_social_merged.csv`
- `tabular_model_features.csv`

To run the standalone preprocessing script:

```bash
pip install -r requirements.txt
python src/preprocess_tabular.py
```

### Facial image data

The image notebook uses the group's collected images. Each participant has neutral, smiling and surprised expressions. The notebook checks the files, identifies participant and expression labels, detects faces and resizes the face crops to 128 × 128 pixels.

I applied image augmentation using small rotations, horizontal flipping and contrast normalization. HOG features were then extracted and saved in `data/images/processed/image_features.csv`.

The participant labels used in the project are:

| Participant | ID |
|---|---|
| Ella | participant_01 |
| King | participant_02 |
| Nicole | participant_03 |
| Sarah | participant_04 |

There were 12 original images. Face detection succeeded for 11 images, while one image was skipped. The augmented feature table contained 55 rows. Augmented copies are variations of the original images, not additional independent samples.

### Voice recordings

The audio notebook processes two phrases from each participant: “Yes, approve” and “Confirm transaction”. It checks the recordings, converts them to mono and resamples them to 16 kHz.

I used pitch shifting, time stretching and added low-level noise to create augmented versions of the recordings. The feature extraction step calculates MFCC statistics, spectral roll-off and RMS energy. The results are saved in `data/audio/processed/audio_features.csv`.

The notebook found eight original WAV recordings, with two recordings for each of the four participants. It created 48 rows in the augmented audio feature table. These augmented rows are not additional independent voice samples.

## Model training and evaluation

The fourth notebook trains and compares models for all three tasks.

- **Facial identity:** Linear SVM and Random Forest using HOG features.
- **Voice identification:** RBF SVM and Random Forest using audio features.
- **Product category:** Logistic Regression and Random Forest using customer features.

The notebook reports accuracy, precision, recall, F1-score and log loss where probability estimates are available. It also produces classification reports and confusion matrices.

The face and voice test sets are very small, with only four held-out original samples each. Both models achieved perfect scores on these small test sets, but that does not show that they would work reliably with new people or recording conditions.

The product-category models achieved 22.58% accuracy on a five-class task. This is only slightly above the 20% random-choice baseline, so the product predictions are weak and should be treated as experimental.

The notebook saves the trained models in `models/` and the evaluation tables in `data/processed/model_evaluation/`.

## Transaction simulation

The command-line simulation uses the saved feature tables and trained models. It asks the operator to choose:

1. The participant account being claimed.
2. The participant whose held-out face sample is presented.
3. The participant whose held-out voice sample is presented.

The system compares the predicted face and voice identities with the claimed account. The transaction is rejected if either identity does not match. If both match, the product-category model provides an experimental category suggestion.

Run the simulation from the project root after running the preprocessing and model-training notebooks:

```bash
python src/transaction_simulation.py
```

For the demonstration, test one matching face-and-voice combination and two mismatches: one where the face does not match the claimed account, and another where the voice does not match.

This simulation uses recorded feature rows from the notebooks. It does not capture a live camera image or microphone recording. It also has no liveness detection or anti-spoofing, so it must not be used to authorize real transactions.

## Main limitations

The biggest limitation is the small amount of biometric data. There are only four participants, three original images per participant and two original voice recordings per participant. Augmentation increases the number of training rows, but it does not replace collecting more genuine samples.

The product-category model also performed poorly. More customer data, a review of the customer-ID matching assumption and further feature experiments would be needed to improve it.

The results should therefore be understood as a demonstration of the preprocessing and modelling workflow rather than a finished biometric or recommendation system.

## Running the notebooks

Install the packages listed in `requirements.txt`, then run the notebooks in this order:

1. `01_tabular_preprocessing.ipynb`
2. `02_image_preprocessing.ipynb`
3. `03_audio_preprocessing.ipynb`
4. `04_model_training_evaluation.ipynb`

The transaction simulation can be run after the models and processed datasets have been generated.

## Data and consent

The facial images and voice recordings were collected from group participants who agreed to their use for this assignment. These files contain biometric information, so access should remain limited to the people who need them for the coursework. They should not be reused for another purpose without permission.

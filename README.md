# Diabetes Prediction App

A machine-learning project that estimates a diabetes outcome from eight patient-provided measurements. The model is trained with **Logistic Regression** and served through a simple **Streamlit** web interface.

> **Important:** This is an educational machine-learning project, not a medical device. Predictions and probabilities are not a diagnosis and must not be used to make treatment decisions. Consult a qualified healthcare professional for medical advice and testing.

## Project overview

The project uses the diabetes dataset included in this repository. The notebook explores the data, prepares the features, trains and evaluates a Logistic Regression classifier, and saves the fitted model and feature scaler. The Streamlit app loads those saved artifacts and produces a prediction from values entered in the browser.

## Features

- Exploratory data analysis with summary statistics, histograms, box plots, a correlation heatmap, and a pair plot.
- Eight input features: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age.
- `StandardScaler` preprocessing to standardize model inputs.
- Logistic Regression classification.
- Evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Streamlit interface that displays the predicted class and the model's estimated probability for the positive class.
- Saved model and scaler files loaded by the application at runtime.

## Dataset

The provided `diabetes.csv` file contains **768 rows** and **9 columns**: eight predictor features and the `Outcome` target.

| Column | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Glucose measurement |
| `BloodPressure` | Blood pressure measurement |
| `SkinThickness` | Skin thickness measurement |
| `Insulin` | Insulin measurement |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function value |
| `Age` | Age |
| `Outcome` | Target label: `0` = No Diabetes, `1` = Diabetes |

The dataset contains 500 records with outcome `0` and 268 records with outcome `1`. The notebook reports no null values in the supplied CSV. A zero in some measurement columns may represent a recorded value in this file; review the dataset's data dictionary and preprocessing assumptions before using it for any serious analysis.

## Model training and evaluation

The notebook uses an 80/20 train-test split with `random_state=42` and stratification on the target. It fits `StandardScaler` on the training features and applies that fitted scaler to the test features, then trains `LogisticRegression(random_state=42)`.

The notebook records the following results on its held-out test split:

| Metric | Score |
|---|---:|
| Accuracy | 0.714 |
| Precision | 0.609 |
| Recall | 0.519 |
| F1-score | 0.560 |
| ROC-AUC | 0.823 |

These are the values recorded in the notebook, not a guarantee of future performance. They depend on the supplied dataset, split, preprocessing, and model configuration.

## Repository structure

The Streamlit script expects the two serialized artifacts to be in the same directory as the app script.

```text
.
├── app.py
├── diabetes.csv
├── logistic regression  (1).ipynb
├── logistic_regression_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

If your repository uses a different notebook filename, folder structure, or app filename, adjust this tree to match the files actually committed. Keep both `.pkl` files alongside `app.py` unless you also update the paths in the application.

## Installation

Python 3.9 or newer is recommended. From the project directory, create and activate a virtual environment:

**Windows (Command Prompt)**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The requirements used by the project are:

```text
streamlit
scikit-learn
numpy
pandas
joblib
```

## Run the application

Make sure these files are in the same directory:

- `app.py` (rename the uploaded `app(2).py` to `app.py`, if needed)
- `logistic_regression_model.pkl`
- `scaler.pkl`

Then run:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your web browser, enter the eight feature values, and select **Predict**.

## How prediction works

1. The app collects the eight feature values in the same order used during training.
2. The saved `StandardScaler` transforms the input values.
3. The saved Logistic Regression model predicts the class.
4. The app displays `Diabetes` or `No Diabetes` and `predict_proba` for class `1` as a percentage.

The displayed percentage is a model-estimated probability for this trained classifier; it is not an individual's clinically validated risk.

## Technologies

- Python
- Pandas and NumPy
- scikit-learn (`StandardScaler`, `LogisticRegression`, model evaluation metrics)
- Matplotlib and Seaborn (notebook visualizations)
- Joblib (model persistence)
- Streamlit (interactive web app)

## Possible improvements

- Add input validation and more careful handling of physiologically implausible values.
- Use a scikit-learn `Pipeline` so preprocessing and classification are stored together.
- Add cross-validation and calibration analysis, and evaluate sensitivity/specificity and threshold trade-offs.
- Document the dataset's provenance, license, intended population, and limitations.
- Add automated tests and a reproducible environment with pinned dependency versions.

## Disclaimer

For learning and demonstration only. This project is not validated for clinical use, does not replace laboratory testing or professional medical evaluation, and should not be used to diagnose diabetes or guide treatment.

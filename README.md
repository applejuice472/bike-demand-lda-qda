# Bike demand classification with LDA and QDA

Predicting whether Capital Bikeshare in Washington D.C. needs more bikes in a given hour, using linear and quadratic discriminant analysis evaluated with nested cross-validation.

This is my part of a group project in Statistical Machine Learning at Uppsala University (2026). The group compared five model families; this repository contains the discriminant analysis part, which I was responsible for.

## The problem

Each of the 1,600 observations is one hour, described by time (hour, weekday, month, holiday, summertime) and weather (temperature, humidity, precipitation, wind and more). The target is whether demand was high or low.

The data is imbalanced: about 82% of hours are low demand. A model that always says "low" gets 82% accuracy while being useless, so **F1 for the high demand class** is the main metric.

## Method

- **Encoding:** hour, month and weekday are one-hot encoded with `drop_first=True`. Keeping all dummy columns makes the covariance matrices singular, which crashes QDA and destabilises LDA.
- **Scaling:** continuous features are standardised inside a `Pipeline`, so the scaler is fitted on each training fold only and never sees test data.
- **Nested cross-validation:** an inner 5-fold CV tunes hyperparameters (LDA solver and shrinkage, QDA regularisation), and an outer 5-fold CV estimates performance on data the tuning never saw. All folds are stratified to keep the class balance.
- **Baselines:** four naive predictors give a floor that the models have to beat.

## Results

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Always low demand | 0.820 | 0.000 | 0.000 | 0.000 |
| Always high demand | 0.180 | 0.305 | 0.180 | 1.000 |
| Random 50/50 | 0.492 | 0.259 | 0.176 | 0.493 |
| Stratified random | 0.706 | 0.214 | 0.206 | 0.222 |
| QDA (nested CV) | 0.826 | 0.596 | 0.514 | 0.712 |
| **LDA (nested CV)** | **0.877** | **0.631** | **0.686** | **0.587** |

LDA roughly doubles the F1 of the best baseline. QDA catches more high demand hours (higher recall) but raises far more false alarms. With 52 features and only about 290 high demand samples, estimating a separate covariance matrix per class is too much for the data, so QDA overfits and needs heavy regularisation. LDA's shared covariance matrix, with automatic shrinkage, generalises better here.

The numbers differ slightly from the course report because scaling now happens inside each fold, which removes a small source of leakage.

## Run it

```bash
pip install -r requirements.txt
python lda_qda_nested_cv.py --data data/training_data_VT2026.csv
```

The dataset (`training_data_VT2026.csv`) was provided by the course and is not included here. Place it in `data/` before running.

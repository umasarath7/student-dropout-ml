# Predicting Student Dropout and Academic Success

**ML Assignment 2, M.Tech (DSE), Machine Learning**

- **Live app:** `<paste your Streamlit app link here>`
- **GitHub repo:** `<paste your GitHub repository link here>`

---

## a. Problem statement

Many students who start a degree never finish it. If a university can spot these students early, for example right after the first year, it can step in with mentoring, fee support or academic help before it is too late.

The goal of this project is to predict, from information available by the end of a student's first year, which of three outcomes a student will reach:

| Class | Meaning |
|---|---|
| **Dropout** | Left the course before finishing |
| **Enrolled** | Still studying at the end of the normal course length (took longer than expected) |
| **Graduate** | Finished the course on time |

This is a **multi-class classification** problem. I trained five different classifiers on the same data and compared them with six evaluation metrics. The best model is shown in an interactive Streamlit web app.

---

## b. Dataset description

- **Name:** Predict Students' Dropout and Academic Success
- **Source:** UCI Machine Learning Repository (dataset id 697), also published on Kaggle as *"Higher Education Predictors of Student Retention"*
- **Origin:** Records from a higher-education institution in Portugal, covering students in different undergraduate degrees (agronomy, design, education, nursing, journalism, management, social service, technologies and more)
- **Size:** 4,424 students (rows), 34 input features + 1 target column
- **Missing values:** none. **Duplicate rows:** none.

**Target distribution (the classes are imbalanced):**

| Class | Students | Share |
|---|---|---|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |

**What the 34 features describe:**

| Group | Example columns |
|---|---|
| Personal background | Marital status, Gender, Age at enrollment, Nationality, Displaced, Educational special needs, International |
| How they joined | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification |
| Family | Mother's and Father's qualification, Mother's and Father's occupation |
| Money | Debtor, Tuition fees up to date, Scholarship holder |
| First-year academics | Curricular units in semester 1 and 2: credited, enrolled, evaluations, approved, average grade, without evaluations |
| Economy at entry | Unemployment rate, Inflation rate, GDP |

### How I prepared the data

1. **Train/test split:** 80% for training (3,539 students) and 20% for testing (885 students). The split is *stratified*, so both parts keep the same class proportions. The test part is saved as `test_data.csv` and no model sees it during training.
2. **Coded columns become one-hot columns.** Columns like `Course` or `Mother's occupation` store category codes (1, 2, 3...), not real amounts. A code of 12 is not "bigger" than a code of 3, so these nine columns are one-hot encoded. Very rare codes (fewer than 20 students) are grouped together so they don't add noise.
3. **Numbers are standardised** (mean 0, standard deviation 1). This matters a lot for kNN, SVM and Logistic Regression, which are sensitive to scale.
4. **Four new features I added:**
   - `Pass rate 1st sem` = units approved ÷ units enrolled (semester 1)
   - `Pass rate 2nd sem` = the same for semester 2
   - `Grade change` = semester 2 average grade minus semester 1 average grade
   - `Units approved total` = approved units across both semesters

   The idea is that *what share* of subjects a student passes says more than the raw count, because different courses have different loads. This paid off: `Pass rate 2nd sem` became the single most important feature in the decision tree (about 65% of its total importance).
5. **Tuning:** each model's main settings were chosen with 5-fold cross-validation on the training set only, using macro F1 as the score.

All of this lives inside one scikit-learn `Pipeline` per model. The saved model therefore takes the raw CSV columns and does every step itself, so the app never has to repeat the preprocessing by hand.

---

## c. GitHub repository

**Link:** `<paste your GitHub repository link here>`

```
project-folder/
├── app.py                     Streamlit web app
├── requirements.txt           Python packages (pinned versions)
├── README.md                  This file
├── test_data.csv              The 20% hold-out test set (885 rows, with Target)
├── data/
│   └── students.csv           Full original dataset (4,424 rows)
└── model/
    ├── features.py            Loading, feature engineering, encoding and scaling
    ├── metrics.py             The six evaluation metrics
    ├── logistic_regression.py Model 1 definition and tuning grid
    ├── decision_tree.py       Model 2
    ├── knn.py                 Model 3
    ├── naive_bayes.py         Model 4
    ├── svm.py                 Model 5
    ├── train_all.py           Trains, tunes, evaluates and saves all five
    └── saved/
        ├── logistic_regression.joblib
        ├── decision_tree.joblib
        ├── knn.joblib
        ├── naive_bayes.joblib
        ├── svm.joblib
        ├── results.csv        Comparison table below
        └── best_params.json   Settings picked by cross-validation
```

**Run it yourself:**

```bash
pip install -r requirements.txt
python -m model.train_all      # retrains the models and rewrites test_data.csv
streamlit run app.py           # opens the app at http://localhost:8501
```

---

## d. Models used

All five models were trained on the same 3,539 students and tested on the same 885 unseen students.

| Model | Settings chosen by cross-validation |
|---|---|
| Logistic Regression | C = 5.0 (multinomial, L2 penalty) |
| Decision Tree | criterion = entropy, max_depth = 6, min_samples_leaf = 5 |
| kNN | k = 5 neighbours, distance-weighted votes |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 |
| Support Vector Machine | RBF kernel, C = 3.0, gamma = "scale", sigmoid-calibrated for probabilities |

### A note on the metrics

Because there are three classes and they are not balanced:

- **Precision, Recall and F1** are **macro averages**: they are worked out for each class separately and then averaged, so the small *Enrolled* group counts just as much as the big *Graduate* group. This is stricter than accuracy, which is why these numbers come out lower.
- **AUC** is the **one-vs-rest** ROC AUC, averaged over the three classes.
- **MCC** (Matthews Correlation Coefficient) uses the whole confusion matrix. It runs from -1 to +1, where 0 means no better than guessing. It is a good single number when classes are imbalanced.

### Comparison table (test set, 885 students)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7808 | **0.8865** | 0.7346 | 0.6950 | 0.7066 | 0.6365 |
| Decision Tree | 0.7503 | 0.8556 | 0.6985 | 0.6783 | 0.6854 | 0.5890 |
| kNN | 0.7379 | 0.8197 | 0.6868 | 0.6396 | 0.6513 | 0.5642 |
| Naive Bayes | 0.7153 | 0.8187 | 0.6725 | 0.6375 | 0.6471 | 0.5313 |
| Support Vector Machine | **0.7853** | 0.8568 | **0.7468** | **0.7056** | **0.7192** | **0.6450** |

**Recall for each class (how many students of each type the model finds):**

| Model | Dropout | Enrolled | Graduate |
|---|---|---|---|
| Logistic Regression | 0.761 | 0.390 | 0.934 |
| Decision Tree | 0.690 | **0.447** | 0.898 |
| kNN | 0.683 | 0.308 | 0.928 |
| Naive Bayes | 0.623 | 0.403 | 0.887 |
| Support Vector Machine | 0.757 | 0.428 | 0.932 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | A very strong result for the simplest model. It is a close second on every metric and has the **best AUC (0.887)**, so it ranks students by risk better than any other model. Its training and test accuracy are close (0.80 vs 0.78), so it is not overfitting. Most of the signal in this data is roughly linear (more subjects passed means more likely to graduate), which suits this model. Its weak spot is the *Enrolled* class: it finds only 39% of those students and tends to label them Graduate. Its coefficients are also easy to explain to non-technical staff. |
| **Decision Tree** | Middle of the table (accuracy 0.75, MCC 0.59). Cross-validation picked a fairly shallow tree (depth 6), which kept overfitting under control (train 0.79 vs test 0.75). Interestingly it has the **best recall for *Enrolled* (0.45)**, because its hard if-then splits can separate "passing some subjects but not all" students. Its rules are easy to read, and it showed that `Pass rate 2nd sem` alone drives about 65% of its decisions, followed by `Tuition fees up to date`. It trails the top models because a single tree makes blocky, unstable decisions. |
| **kNN** | Second weakest (F1 0.65). It fits the training data perfectly (train accuracy 1.00 with distance weighting) but drops to 0.74 on the test set, which is a clear sign it is memorising rather than generalising. With about 60 columns after one-hot encoding, distances between students become less meaningful, and the many 0/1 columns blur who counts as a "neighbour". It also has the **lowest recall for *Enrolled* (0.31)**: these students sit close to both Graduates and Dropouts, so their neighbours usually outvote them. |
| **Naive Bayes** | The weakest overall (accuracy 0.72, MCC 0.53), as expected. It assumes every feature is independent, but here many features are strongly linked: semester 1 and semester 2 grades, enrolled vs approved units, and mother's vs father's education. It ends up counting the same evidence several times. It also assumes a bell-shaped distribution for each feature, which does not fit 0/1 columns or counts. It needed heavy smoothing (0.1) to work at all. On the plus side it trains instantly and still beats guessing by a wide margin (AUC 0.82). |
| **Support Vector Machine** | **The best model overall.** It is top on Accuracy (0.785), Precision, Recall, F1 (0.719) and MCC (0.645). The RBF kernel lets it draw curved boundaries, which helps with the hard *Enrolled* students in the middle (recall 0.43 vs 0.39 for Logistic Regression). Its AUC (0.857) is lower than Logistic Regression's because SVM does not produce probabilities directly and they have to be estimated by an extra calibration step, which slightly blurs the ranking. There is a mild train/test gap (0.90 vs 0.79), and it is the slowest of the five to train. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** wins: it has the highest Accuracy, macro F1 and MCC, which are the fairest measures for imbalanced classes. **Logistic Regression is a very close runner-up** and would be a sensible choice if the university needs a model it can easily explain, or needs well-ranked risk scores (best AUC). Every model finds *Graduate* (about 90%+) and *Dropout* (62–76%) well, but all of them struggle with *Enrolled* (31–45%). That makes sense: these students are partly succeeding and partly not, so they look like a mix of the other two groups. For a real early-warning system, the most useful next steps would be class weighting or oversampling for the *Enrolled* group, and trying ensemble methods such as Random Forest or Gradient Boosting. |

---

## Streamlit app

**Live link:** `<paste your Streamlit app link here>`

What the app offers:

1. **CSV upload** (sidebar). Upload `test_data.csv`, or any CSV with the same columns. A *Download sample* button gives you the right file, and you can also tick *use the bundled test set* to try the app straight away.
2. **Model selection dropdown** to switch between the five models.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
4. **Confusion matrix** (heatmap) **and classification report** (precision, recall and F1 for each class), plus a note on which class the model finds hardest.
5. **Compare all models** tab: all five models scored side by side on the uploaded data, with the best value in each column highlighted and a bar chart you can switch between metrics.
6. **Predictions** tab: the predicted outcome and the probability of each class for every student, with a filter for students predicted to drop out and a CSV download.

If the uploaded file has no `Target` column, the app still makes predictions and simply skips the scores.

### Deploying on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app**, pick the repository and the `main` branch, and set the main file to `app.py`.
4. Click **Deploy**.

The package versions in `requirements.txt` are pinned to the exact versions used for training. The saved `.joblib` models only load correctly with the same scikit-learn version, so please don't remove the pins.

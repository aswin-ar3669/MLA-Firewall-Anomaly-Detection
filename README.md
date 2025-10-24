🧠 Machine Learning Algorithms for Firewall Anomaly Detection

This project focuses on building, training, and evaluating multiple machine learning models to detect anomalies in firewall network traffic logs. It compares the performance of supervised and unsupervised learning algorithms using configurable cross-validation strategies.

🚀 Key Features

Comprehensive ML Pipeline: Train, validate, and predict firewall anomalies with different algorithms.

Supervised & Unsupervised Comparison: Evaluate both approaches on the same dataset.

Cross-Validation Support:

Stratified K-Fold (default, 5 folds)

Time-aware CV (for chronological datasets)

Multiple Random Trials: Run multiple randomized trials (default: 40) for robust evaluation.

Flexible Training CLI: Easily customize iterations, folds, and output directories.

Model Benchmarking: Compare accuracy, precision, recall, F1-score, and AUC across algorithms.

📁 Project Structure

firewall-anomaly-detection/

│

├── data/

│ ├── firewall\_train.csv # Training dataset

│ ├── firewall\_test.csv # Optional test dataset

│

├── models/

│ ├── supervised/ # Saved supervised models

│ ├── unsupervised/ # Saved unsupervised models

│

├── scripts/

│ ├── train\_model.py # Main training script

│ ├── predict.py # Script to make predictions

│ ├── evaluate.py # Script to compare results

│

├── results/

│ ├── metrics\_summary.csv # Evaluation metrics

│ ├── confusion\_matrix.png

│ ├── roc\_curves.png

│

└── README.md

🧩 Supported Algorithms

🔹 Supervised Learning

Logistic Regression

Random Forest

Gradient Boosting (XGBoost / LightGBM)

Support Vector Machines (SVM)

Neural Networks (MLPClassifier)

🔹 Unsupervised Learning

Isolation Forest

One-Class SVM

DBSCAN

Autoencoder (if using deep learning backend)

⚙️ Installation

\# Create virtual environment

python -m venv venv

source venv/bin/activate # or venv\\Scripts\\activate on Windows

\# Install dependencies

pip install -r requirements.txt

🧪 Usage

1️⃣ Default Training (Stratified K-Fold CV, 5 folds, 40 trials per model)

python ./train\_model.py --train\_csv ./data/firewall\_train.csv --outdir ./models

2️⃣ Time-Aware Cross-Validation (Chronological Data)

python ./train\_model.py --train\_csv ./data/firewall\_train.csv --outdir ./models --cv time

3️⃣ Faster Dry Run (Fewer Iterations)

python ./train\_model.py --train\_csv ./data/firewall\_train.csv --outdir ./models --n\_iter 10

4️⃣ Test with Saved Model

python test_model.py --test_csv ./firewall_test.csv --iso_model ./models/model_isolation_forest.joblib --clf_model ./models/model_gb_classifier.joblib --outdir ./models/Visualization

4️⃣ Predict with Saved Model

python ./predict.py --input_csv ./firewall_test.csv --iso_model ./models/model_isolation_forest.joblib --clf_model ./models/model_gb_classifier.joblib --out ./models/result/predictions.csv

5️⃣ Compare Supervised vs Unsupervised

1. Supervised Learning

python compare_supervised.py --csv firewall_train.csv --outdir ./models/results

2. UnSupervised Learning 

python compare_unsupervised.py --csv firewall_train.csv --outdir ./models/results

📊 Evaluation Metrics

MetricDescription

AccuracyCorrect predictions ratio

Precision% of predicted anomalies that are true

Recall (Sensitivity) % of actual anomalies correctly identified

F1-ScoreHarmonic mean of Precision & Recall

AUC-ROCDiscrimination ability of classifier

Visual outputs:

Confusion Matrix

ROC Curve

Precision-Recall Curve

🧠 Example Output

Model: RandomForestClassifier

\---------------------------------------

Accuracy: 0.978

Precision: 0.965

Recall: 0.981

F1-Score: 0.972

AUC: 0.991

\---------------------------------------

🧾 Configuration Options

ArgumentDescriptionDefault

\--train\_csvPath to training CSV file./firewall\_train.csv

\--outdirOutput directory for models./models

\--cvCross-validation type (kfold, time)kfold

\--n\_iterNumber of random trials40

\--foldsNumber of folds for K-Fold CV5

\--seedRandom seed for reproducibility42

🧮 Data Preprocessing

Load and clean firewall logs (firewall\_train.csv)

Encode categorical variables (IP, ports, protocol types, etc.)

Normalize numerical features

Handle missing values and outliers

Split into train/test sets (if not pre-split)

🧰 Dependencies

numpy

pandas

scikit-learn

matplotlib

xgboost

lightgbm

joblib

seaborn

🧩 Future Improvements

Integration with Deep Autoencoders for advanced anomaly detection

Real-time firewall log stream monitoring

Model explainability with SHAP

Integration with Streamlit Dashboard for visualization

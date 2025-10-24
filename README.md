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
│   ├── firewall_train.csv      # Training dataset
│   ├── firewall_test.csv       # Optional test dataset
│
├── models/
│   ├── supervised/             # Saved supervised models
│   ├── unsupervised/           # Saved unsupervised models
│
├── scripts/
│   ├── train_model.py          # Main training script
│   ├── predict.py              # Script to make predictions
│   ├── evaluate.py             # Script to compare results
│
├── results/
│   ├── metrics_summary.csv     # Evaluation metrics
│   ├── confusion_matrix.png
│   ├── roc_curves.png
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
# Clone the repository
git clone https://github.com/yourusername/firewall-anomaly-detection.git
cd firewall-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

🧪 Usage
1️⃣ Default Training (Stratified K-Fold CV, 5 folds, 40 trials per model)
python ./train_model.py --train_csv ./data/firewall_train.csv --outdir ./models

2️⃣ Time-Aware Cross-Validation (Chronological Data)
python ./train_model.py --train_csv ./data/firewall_train.csv --outdir ./models --cv time

3️⃣ Faster Dry Run (Fewer Iterations)
python ./train_model.py --train_csv ./data/firewall_train.csv --outdir ./models --n_iter 10

4️⃣ Predict with Saved Model
python ./predict.py --model ./models/supervised/random_forest.pkl --test_csv ./data/firewall_test.csv --outdir ./results

5️⃣ Compare Supervised vs Unsupervised
python ./evaluate.py --results_dir ./results --compare supervised unsupervised

📊 Evaluation Metrics
Metric	Description
Accuracy	Correct predictions ratio
Precision	% of predicted anomalies that are true
Recall (Sensitivity)	% of actual anomalies correctly identified
F1-Score	Harmonic mean of Precision & Recall
AUC-ROC	Discrimination ability of classifier

Visual outputs:

Confusion Matrix

ROC Curve

Precision-Recall Curve

🧠 Example Output
Model: RandomForestClassifier
---------------------------------------
Accuracy: 0.978
Precision: 0.965
Recall: 0.981
F1-Score: 0.972
AUC: 0.991
---------------------------------------

🧾 Configuration Options
Argument	Description	Default
--train_csv	Path to training CSV file	./firewall_train.csv
--outdir	Output directory for models	./models
--cv	Cross-validation type (kfold, time)	kfold
--n_iter	Number of random trials	40
--folds	Number of folds for K-Fold CV	5
--seed	Random seed for reproducibility	42
🧮 Data Preprocessing

Load and clean firewall logs (firewall_train.csv)

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

# Default: Stratified K-Fold CV (5 folds), 40 random trials per model
    # python ./train_model.py --train_csv ./firewall_train.csv --outdir ./models
# Time-aware CV (if rows are chronological)
    # python train_model.py --train_csv ./firewall_train.csv --outdir ./models --cv time
# Faster dry run
    # python train_model.py --train_csv ./firewall_train.csv --outdir ./models --n_iter 10
import argparse
import json
import os
from datetime import datetime
from ipaddress import ip_address, IPv6Address, ip_network

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Feature utilities
# -----------------------------
PRIVATE_NETS = [
    ip_network('10.0.0.0/8'),
    ip_network('172.16.0.0/12'),
    ip_network('192.168.0.0/16'),
    ip_network('169.254.0.0/16'),
]

def ip_to_int_safe(val):
    try:
        if pd.isna(val):
            return np.nan
        ip = ip_address(str(val))
        if isinstance(ip, IPv6Address):
            return int(ip) % (2**32)
        return int(ip)
    except Exception:
        return np.nan

def is_private_ip(val):
    try:
        if pd.isna(val):
            return False
        ip = ip_address(str(val))
        return any(ip in net for net in PRIVATE_NETS)
    except Exception:
        return False

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist
    for col in ["Date", "Time"]:
        if col not in df.columns:
            df[col] = None

    # Parse datetime parts
    def parse_dt(row):
        d, t = row.get("Date"), row.get("Time")
        if pd.isna(d) or pd.isna(t):
            return None
        for fmt in ("%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(f"{d} {t}", fmt)
            except Exception:
                continue
        return None

    dt_series = df.apply(parse_dt, axis=1)
    df["hour"] = dt_series.apply(lambda x: x.hour if isinstance(x, datetime) else np.nan)
    df["minute"] = dt_series.apply(lambda x: x.minute if isinstance(x, datetime) else np.nan)
    df["weekday"] = dt_series.apply(lambda x: x.weekday() if isinstance(x, datetime) else np.nan)

    # IP engineering
    for col in ["Source", "Destination"]:
        if col not in df.columns:
            df[col] = None

    df["Source_ip_int"] = df["Source"].apply(ip_to_int_safe)
    df["Dest_ip_int"] = df["Destination"].apply(ip_to_int_safe)
    df["src_private"] = df["Source"].apply(is_private_ip).astype(int)
    df["dst_private"] = df["Destination"].apply(is_private_ip).astype(int)

    # Add any missing numeric/categorical columns
    numeric_cols = ["Source_port", "Dest_Port", "Duration", "Bytes", "Packets",
                    "hour", "minute", "weekday", "Source_ip_int", "Dest_ip_int",
                    "src_private", "dst_private"]
    categorical_cols = ["Protocol", "Service", "State", "Action", "Policy_ID"]

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = None

    # Coerce numerics
    for c in ["Source_port", "Dest_Port", "Duration", "Bytes", "Packets"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def build_preprocessor():
    numeric_features = ["Source_port", "Dest_Port", "Duration", "Bytes", "Packets",
                        "hour", "minute", "weekday", "Source_ip_int", "Dest_ip_int",
                        "src_private", "dst_private"]
    categorical_features = ["Protocol", "Service", "State", "Action", "Policy_ID"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def build_pipelines():
    pre = build_preprocessor()
    iso = IsolationForest(random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=42)

    iso_pipe = Pipeline([("pre", pre), ("iso", iso)])
    gb_pipe  = Pipeline([("pre", pre), ("gb", gb)])
    return iso_pipe, gb_pipe

# --- Custom scorer for Isolation Forest (used directly in RandomizedSearchCV) ---
def auroc_from_iso(estimator, X, y):
    # y must be binary: 1 = attack, 0 = benign
    pre_X = estimator.named_steps["pre"].transform(X)
    # ensure dense for score_samples
    try:
        pre_X = pre_X.toarray()
    except AttributeError:
        pass
    scores = -estimator.named_steps["iso"].score_samples(pre_X)
    if len(np.unique(y)) < 2:
        # if a CV fold has only one class, return 0.5 to avoid failure
        return 0.5
    return roc_auc_score(y, scores)

def main():
    parser = argparse.ArgumentParser(description="Train tuned IsolationForest and GradientBoosting pipelines with RandomizedSearchCV; save models and training summary.")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV (must include Attack column).")
    parser.add_argument("--outdir", required=True, help="Where to save models and summary.")
    parser.add_argument("--cv", choices=["stratified", "time"], default="stratified", help="Cross-validation strategy.")
    parser.add_argument("--n_iter", type=int, default=40, help="RandomizedSearch iterations per model.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    df = generate_features(df)

    if "Attack" not in df.columns or not df["Attack"].notna().any():
        raise SystemExit("Training set must include a non-empty 'Attack' column.")

    # Labels
    y_binary = (df["Attack"].astype(str) != "benign").astype(int)  # for AUROC
    y_multi  = df["Attack"].astype(str)                            # for GB classifier
    X = df  # Pipelines will select required columns

    # CV strategies
    if args.cv == "time":
        n_splits = 5
        cv_iso = TimeSeriesSplit(n_splits=n_splits)
        cv_gb  = TimeSeriesSplit(n_splits=n_splits)
        splitter_info = f"TimeSeriesSplit(n_splits={n_splits})"
    else:
        cv_iso = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y_binary)), y_binary)
        cv_gb  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y_multi)), y_multi)
        splitter_info = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"

    # Build pipelines
    iso_pipe, gb_pipe = build_pipelines()

    # Search spaces
    iso_param_distributions = {
        "iso__n_estimators": [200, 300, 400, 600, 800, 1000],
        "iso__max_samples":  ["auto", 256, 512, 1024, 2048, 4096],
        "iso__max_features": [0.6, 0.8, 1.0],
        "iso__contamination": [0.02, 0.05, 0.1, 0.12, 0.15, 0.2],
        "iso__bootstrap": [False, True],
    }
    gb_param_distributions = {
        "gb__n_estimators": [100, 200, 300, 400, 600],
        "gb__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
        "gb__max_depth": [2, 3, 4, 5],
        "gb__subsample": [0.6, 0.8, 1.0],
        "gb__min_samples_split": [2, 10, 20, 50],
        "gb__min_samples_leaf": [1, 2, 4, 10],
    }

    # Randomized searches (CV included)
    iso_search = RandomizedSearchCV(
        estimator=iso_pipe,
        param_distributions=iso_param_distributions,
        n_iter=args.n_iter,
        scoring=auroc_from_iso,
        cv=cv_iso,
        refit=True,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    gb_search = RandomizedSearchCV(
        estimator=gb_pipe,
        param_distributions=gb_param_distributions,
        n_iter=args.n_iter,
        scoring="f1_macro",
        cv=cv_gb,
        refit=True,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    # Fit
    iso_search.fit(X, y_binary)
    gb_search.fit(X, y_multi)

    # Final metrics on full training (for reference; prefer held-out for evaluation)
    iso_best = iso_search.best_estimator_
    pre_X = iso_best.named_steps["pre"].transform(X)
    try:
        pre_X = pre_X.toarray()
    except AttributeError:
        pass
    iso_scores = -iso_best.named_steps["iso"].score_samples(pre_X)
    iso_auroc = roc_auc_score(y_binary, iso_scores)

    gb_best = gb_search.best_estimator_
    gb_preds = gb_best.predict(X)
    gb_acc = accuracy_score(y_multi, gb_preds)
    gb_f1_macro = f1_score(y_multi, gb_preds, average="macro")
    gb_report = classification_report(y_multi, gb_preds, output_dict=True)
    gb_cm = confusion_matrix(y_multi, gb_preds).tolist()

    # Save models with original expected filenames
    iso_path = os.path.join(args.outdir, "model_isolation_forest.joblib")
    gb_path = os.path.join(args.outdir, "model_gb_classifier.joblib")
    dump(iso_best, iso_path)
    dump(gb_best, gb_path)

    # Save training summary
    summary = {
        "cv": splitter_info,
        "isolation_forest": {
            "best_params": iso_search.best_params_,
            "best_cv_score": iso_search.best_score_,
            "train_auroc": iso_auroc,
            "model_path": iso_path,
        },
        "gradient_boosting": {
            "best_params": gb_search.best_params_,
            "best_cv_score_f1_macro": gb_search.best_score_,
            "train_accuracy": gb_acc,
            "train_f1_macro": gb_f1_macro,
            "classification_report": gb_report,
            "confusion_matrix": gb_cm,
            "model_path": gb_path,
        },
    }
    with open(os.path.join(args.outdir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

# python ./predict.py --input_csv ./data/firewall_test.csv --iso_model ./models/model_isolation_forest.joblib --clf_model ./models/model_gb_classifier.joblib --out ./models/result/predictions.csv
import argparse, os, json
import numpy as np
import pandas as pd
from datetime import datetime
from ipaddress import ip_address, IPv6Address, ip_network
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ATTACK_TYPES = ["benign","port_scan","brute_force","dns_tunnel","exfiltration","ddos"]

# ---------- Feature engineering (same as training) ----------
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
    # Ensure needed columns exist
    for col in ["Date","Time","Source","Destination"]:
        if col not in df.columns:
            df[col] = None

    # Parse Date+Time into hour/minute/weekday
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

    # IP features
    df["Source_ip_int"] = df["Source"].apply(ip_to_int_safe)
    df["Dest_ip_int"] = df["Destination"].apply(ip_to_int_safe)
    df["src_private"] = df["Source"].apply(is_private_ip).astype(int)
    df["dst_private"] = df["Destination"].apply(is_private_ip).astype(int)

    # Ensure numeric/categorical columns exist
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
    # Coerce base numerics
    for c in ["Source_port","Dest_Port","Duration","Bytes","Packets"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
# -----------------------------------------------------------

def dense(X):
    try:
        return X.toarray()
    except AttributeError:
        return X

def main():
    ap = argparse.ArgumentParser(description="Predict attack class and anomaly score on a CSV using trained pipelines.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--iso_model", required=True)
    ap.add_argument("--clf_model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics_out", default=None)
    args = ap.parse_args()

    # Load and engineer features
    df_raw = pd.read_csv(args.input_csv).drop_duplicates().reset_index(drop=True)
    df = generate_features(df_raw.copy())

    # Load fitted pipelines
    iso_pipe = load(args.iso_model)   # expected steps: 'pre' + 'iso'
    clf_pipe = load(args.clf_model)   # expected steps: 'pre' + 'gb'

    # Get preprocessors (accept either 'pre' or legacy 'prep')
    pre_iso = iso_pipe.named_steps.get("pre") or iso_pipe.named_steps.get("prep")
    iso_model = iso_pipe.named_steps.get("iso")
    if pre_iso is None or iso_model is None:
        raise SystemExit("Isolation Forest pipeline must include a preprocess step named 'pre' (or 'prep') and an 'iso' step.")

    # Anomaly score using Isolation Forest
    Z = dense(pre_iso.transform(df))
    anomaly_score = -iso_model.score_samples(Z)

    # Attack predictions using full classifier pipeline
    pred_attack = clf_pipe.predict(df)

    # Save predictions
    out = df_raw.copy()
    out["pred_attack"] = pred_attack
    out["anomaly_score"] = anomaly_score
    out.to_csv(args.out, index=False)
    print("Saved predictions to:", args.out)

    # Optional metrics if labels exist
    if "Attack" in df_raw.columns and df_raw["Attack"].notna().any():
        y_true = df_raw["Attack"].astype(str).values
        y_true_binary = (df_raw["Attack"].astype(str) != "benign").astype(int).values
        try:
            auroc = roc_auc_score(y_true_binary, anomaly_score)
        except Exception:
            auroc = None
        report = classification_report(y_true, pred_attack, digits=4, output_dict=True, zero_division=0)
        labels = ["benign","port_scan","brute_force","dns_tunnel","exfiltration","ddos"]
        cm = confusion_matrix(y_true, pred_attack, labels=labels).tolist()
        acc = report.get("accuracy", None)

        metrics = {
            "anomaly_auroc": auroc,
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": {"labels": labels, "matrix": cm}
        }
        if args.metrics_out is None:
            base, _ = os.path.splitext(args.out)
            args.metrics_out = base + "_metrics.json"
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved metrics to:", args.metrics_out)

if __name__ == "__main__":
    main()

# python test_model.py --test_csv ./firewall_test.csv --iso_model ./model_isolation_forest.joblib --clf_model ./model_gb_classifier.joblib --outdir ./Visualization
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ipaddress import ip_address, IPv6Address, ip_network
from joblib import load
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score
)

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

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def dense(X):
    try:
        return X.toarray()
    except AttributeError:
        return X

def get_steps(pipe):
    pre = pipe.named_steps.get("pre") or pipe.named_steps.get("prep")
    iso = pipe.named_steps.get("iso")
    gb  = pipe.named_steps.get("gb") or pipe.named_steps.get("clf")
    return pre, iso, gb

def get_feature_names_from_pipeline(pipe):
    pre, _, _ = get_steps(pipe)
    if pre is None or not hasattr(pre, "transformers_"):
        return None
    names = []
    for name, transformer, cols in pre.transformers_:
        if name == "remainder":
            continue
        last = list(transformer.named_steps.values())[-1] if hasattr(transformer, "named_steps") else transformer
        if hasattr(last, "get_feature_names_out"):
            try:
                out = last.get_feature_names_out(cols)
                names.extend(list(out))
            except Exception:
                names.extend(list(cols))
        else:
            names.extend(list(cols))
    return names

def plot_confusion_matrix(cm, classes, out_path, title="Confusion matrix"):
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_bar(values, labels, out_path, title, ylabel):
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_hist(values, out_path, title, xlabel, bins=60):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Evaluate models on held-out CSV and generate visualizations.")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--iso_model", required=True)
    ap.add_argument("--clf_model", required=True)
    ap.add_argument("--outdir", default="./")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load data and apply SAME feature engineering as training
    df_raw = pd.read_csv(args.test_csv)
    df = generate_features(df_raw.copy())

    # Load models
    iso_pipe = load(args.iso_model)
    gb_pipe  = load(args.clf_model)

    # ---------- Anomaly scores (Isolation Forest) ----------
    pre_iso, iso_model, _ = get_steps(iso_pipe)
    if pre_iso is None or iso_model is None:
        raise SystemExit("Isolation Forest pipeline must include a preprocess step named 'pre' or 'prep' and an 'iso' step.")
    Z = dense(pre_iso.transform(df))
    anomaly_score = -iso_model.score_samples(Z)

    # ---------- Attack predictions (Gradient Boosting) ----------
    # Use the SAME engineered features for GB as well
    pred_attack = gb_pipe.predict(df)

    # Save predictions CSV
    out_df = df_raw.copy()
    out_df["pred_attack"] = pred_attack
    out_df["anomaly_score"] = anomaly_score
    pred_path = os.path.join(args.outdir, "test_with_predictions.csv")
    out_df.to_csv(pred_path, index=False)

    # ---------- Metrics (if labels exist) ----------
    metrics = {}
    has_labels = "Attack" in df_raw.columns and df_raw["Attack"].notna().any()
    if has_labels:
        y_true_multi = df_raw["Attack"].astype(str).values
        y_true_binary = (df_raw["Attack"].astype(str) != "benign").astype(int).values

        # IF metrics
        auroc = roc_auc_score(y_true_binary, anomaly_score)
        fpr, tpr, _ = roc_curve(y_true_binary, anomaly_score)
        prec, rec, _ = precision_recall_curve(y_true_binary, anomaly_score)
        ap = average_precision_score(y_true_binary, anomaly_score)

        # GB metrics
        acc = accuracy_score(y_true_multi, pred_attack)
        f1m = f1_score(y_true_multi, pred_attack, average="macro")
        report = classification_report(y_true_multi, pred_attack, output_dict=True, zero_division=0)
        labels = sorted(list(pd.unique(np.concatenate([y_true_multi, pred_attack]))))
        if set(ATTACK_TYPES).issubset(set(labels)):
            labels = ATTACK_TYPES
        cm = confusion_matrix(y_true_multi, pred_attack, labels=labels)

        metrics = {
            "isolation_forest": {"auroc": float(auroc), "average_precision": float(ap)},
            "gradient_boosting": {
                "accuracy": float(acc),
                "f1_macro": float(f1m),
                "classification_report": report,
                "labels": labels,
                "confusion_matrix": cm.tolist(),
            },
        }

        with open(os.path.join(args.outdir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # ---------- Plots ----------
        # ROC
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1], linestyle="--")
        plt.title(f"ROC Curve (AUROC={auroc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "roc_curve.png"), bbox_inches="tight")
        plt.close()

        # PR
        plt.figure()
        plt.plot(rec, prec)
        plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "pr_curve.png"), bbox_inches="tight")
        plt.close()

        # Confusion Matrix
        plot_confusion_matrix(cm, labels, os.path.join(args.outdir, "confusion_matrix.png"))

        # Per-class F1
        class_labels, f1_vals = [], []
        for k, v in report.items():
            if k in ["accuracy", "macro avg", "weighted avg"]:
                continue
            class_labels.append(k)
            f1_vals.append(v.get("f1-score", 0.0))
        plot_bar(f1_vals, class_labels, os.path.join(args.outdir, "per_class_f1.png"),
                 "Per-class F1 Scores", "F1-score")

    # Always: anomaly score histogram
    plot_hist(anomaly_score, os.path.join(args.outdir, "anomaly_score_hist.png"),
              "Anomaly Score Distribution", "anomaly_score")

    # Optional: GB feature importances
    try:
        feat_names = get_feature_names_from_pipeline(gb_pipe)
        gb_core = gb_pipe.named_steps.get("gb") or gb_pipe.named_steps.get("clf")
        if gb_core is not None and hasattr(gb_core, "feature_importances_"):
            importances = gb_core.feature_importances_
            idx = np.argsort(importances)[::-1][:25]
            top_names = [feat_names[i] if (feat_names and i < len(feat_names)) else f"feat_{i}" for i in idx]
            top_vals = importances[idx]
            plot_bar(top_vals, top_names, os.path.join(args.outdir, "gb_feature_importance_top25.png"),
                     "Top 25 Feature Importances (Gradient Boosting)", "Importance")
    except Exception as e:
        with open(os.path.join(args.outdir, "viz_warnings.txt"), "a", encoding="utf-8") as f:
            f.write(str(e) + "\n")

    # Console summary
    print("Saved:")
    print(" -", pred_path)
    if metrics:
        print(" -", os.path.join(args.outdir, "test_metrics.json"))
        print(" -", os.path.join(args.outdir, "roc_curve.png"))
        print(" -", os.path.join(args.outdir, "pr_curve.png"))
        print(" -", os.path.join(args.outdir, "confusion_matrix.png"))
        print(" -", os.path.join(args.outdir, "per_class_f1.png"))
    print(" -", os.path.join(args.outdir, "anomaly_score_hist.png"))
    print(" -", os.path.join(args.outdir, "gb_feature_importance_top25.png"))

if __name__ == "__main__":
    main()

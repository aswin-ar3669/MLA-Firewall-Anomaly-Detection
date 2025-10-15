#  python Select_Model.py --csv firewall_train.csv --outdir ./models/results
import argparse, os, json
import numpy as np
import pandas as pd
from datetime import datetime
from ipaddress import ip_address, IPv6Address, ip_network
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# -------- Feature Engineering --------
PRIVATE_NETS = [
    ip_network('10.0.0.0/8'),
    ip_network('172.16.0.0/12'),
    ip_network('192.168.0.0/16'),
    ip_network('169.254.0.0/16'),
]

def ip_to_int_safe(val):
    try:
        if pd.isna(val): return np.nan
        ip = ip_address(str(val))
        if isinstance(ip, IPv6Address):
            return int(ip) % (2**32)
        return int(ip)
    except: return np.nan

def is_private_ip(val):
    try:
        if pd.isna(val): return False
        ip = ip_address(str(val))
        return any(ip in net for net in PRIVATE_NETS)
    except: return False

def parse_dt(d, t):
    try: return datetime.strptime(f"{d} {t}", "%d/%m/%y %H:%M:%S")
    except: return None

def generate_features(df):
    dt = df.apply(lambda r: parse_dt(r.get("Date"), r.get("Time")), axis=1)
    df["hour"] = dt.apply(lambda x: x.hour if isinstance(x, datetime) else np.nan)
    df["minute"] = dt.apply(lambda x: x.minute if isinstance(x, datetime) else np.nan)
    df["weekday"] = dt.apply(lambda x: x.weekday() if isinstance(x, datetime) else np.nan)
    df["Source_ip_int"] = df["Source"].apply(ip_to_int_safe)
    df["Dest_ip_int"] = df["Destination"].apply(ip_to_int_safe)
    df["src_private"] = df["Source"].apply(is_private_ip).astype(int)
    df["dst_private"] = df["Destination"].apply(is_private_ip).astype(int)
    return df

def build_preprocessor(X):
    num_cols = ["Source_port","Dest_Port","Duration","Bytes","Packets","hour","minute","weekday","Source_ip_int","Dest_ip_int","src_private","dst_private"]
    cat_cols = ["Protocol","Service","State","Action","Policy_ID"]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, [c for c in num_cols if c in X.columns]), ("cat", categorical, [c for c in cat_cols if c in X.columns])])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["Attack"].notna()].copy()
    df = generate_features(df)

    X, y = df.drop(columns=["Attack"]), df["Attack"].astype(str)
    pre = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }

    rows = []
    for name, clf in models.items():
        accs, precs, recs, f1s = [], [], [], []
        for tr, te in cv.split(X, y):
            X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            precs.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
            recs.append(recall_score(y_te, y_pred, average="macro", zero_division=0))
            f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        rows.append({"Model": name,"Accuracy": np.mean(accs),"Precision": np.mean(precs),"Recall": np.mean(recs),"F1_macro": np.mean(f1s)})
    out_csv = os.path.join(args.outdir,"supervised_results.csv")
    pd.DataFrame(rows).to_csv(out_csv,index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()

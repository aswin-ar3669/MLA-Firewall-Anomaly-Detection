#  python compare_unsupervised.py --csv firewall_train.csv --outdir ./results
import argparse, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

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

    X, y_multi = df.drop(columns=["Attack"]), df["Attack"].astype(str)
    y_binary = (y_multi != "benign").astype(int)
    pre = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    models = {
        "Isolation Forest": IsolationForest(n_estimators=300, contamination=0.1, random_state=42),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    }

    rows = []
    for name, mdl in models.items():
        aurocs, aps, f1s = [], [], []
        for tr, te in cv.split(X, y_binary):
            X_tr, X_te, y_te = X.iloc[tr], X.iloc[te], y_binary.iloc[te]
            pre_fit = pre.fit(X_tr)
            Z_tr, Z_te = pre_fit.transform(X_tr), pre_fit.transform(X_te)
            mdl.fit(Z_tr)
            if hasattr(mdl,"score_samples"):
                scores = -mdl.score_samples(Z_te)
            else:
                scores = -mdl.decision_function(Z_te)
            aurocs.append(roc_auc_score(y_te,scores))
            aps.append(average_precision_score(y_te,scores))
            prec,rec,_=precision_recall_curve(y_te,scores)
            f1_curve=(2*prec*rec)/(prec+rec+1e-12)
            f1s.append(np.nanmax(f1_curve))
        rows.append({"Model":name,"AUROC":np.mean(aurocs),"AP":np.mean(aps),"BestF1":np.mean(f1s)})
    out_csv=os.path.join(args.outdir,"unsupervised_results.csv")
    pd.DataFrame(rows).to_csv(out_csv,index=False)
    print("Saved:", out_csv)

if __name__=="__main__":
    main()

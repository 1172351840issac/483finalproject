#!/usr/bin/env python3
"""
Run 5‑fold CV on 4 models (LR, SVM, RF, XGB) with / without SMOTE.
Outputs: results/metrics.csv  roc.png  best_model.pkl
"""

import os, argparse, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tgt = next((c for c in df.columns if c.lower().startswith("fraud")), None)
    if tgt is None:
        raise ValueError("No column starting with 'fraud' found.")
    df = df.rename(columns={tgt: "FraudFound"})
    df = df.dropna(axis=1, how="all")           # drop fully‑null cols
    if df["FraudFound"].dtype == object:
        df["FraudFound"] = (
            df["FraudFound"].str.strip().str.upper().map({"Y": 1, "YES": 1, "N": 0, "NO": 0})
        )
    df["FraudFound"] = df["FraudFound"].astype(int)
    return df


def make_preprocessor(df):
    num_cols = df.select_dtypes(include="number").columns.difference(["FraudFound"])
    cat_cols = df.select_dtypes(include="object").columns.difference(["FraudFound"])
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])


def build_pipe(model, sampler, df):
    steps = [("pre", make_preprocessor(df))]
    if sampler == "smote":
        steps.append(("smote", SMOTE()))
    steps.append(("clf", model))
    return ImbPipeline(steps)


def main(path):
    os.makedirs("results", exist_ok=True)
    df = load_data(path)
    X, y = df.drop("FraudFound", axis=1), df["FraudFound"]

    models = {
        "LR":  LogisticRegression(max_iter=200, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True),
        "RF":  RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              subsample=0.9, eval_metric="logloss",
                              tree_method="hist", random_state=42)
    }
    samplers = ["none", "smote"]
    rows = []
    for s in samplers:
        for name, mdl in models.items():
            pipe = build_pipe(mdl, s, df)
            cv = cross_validate(pipe, X, y, scoring={"F1": "f1", "AUC": "roc_auc"},
                                cv=5, n_jobs=-1)
            rows.append({"Model": name, "Sampler": s,
                         "F1": cv["test_F1"].mean(), "AUC": cv["test_AUC"].mean()})
            print(f"✓ {name:3} | {s:<5}  F1={cv['test_F1'].mean():.3f}")

    pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)

    best = max(rows, key=lambda r: r["F1"])
    pipe = build_pipe(models[best["Model"]], best["Sampler"], df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    joblib.dump(pipe, "results/best_model.pkl")
    RocCurveDisplay.from_estimator(pipe, Xte, yte)
    plt.title(f"ROC – {best['Model']} ({best['Sampler']})")
    plt.savefig("results/roc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Artifacts saved → results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/insurance_claims.csv",
                    help="CSV file with fraud_reported column")
    main(ap.parse_args().data)

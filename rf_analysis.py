import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_onehot():
    # Compat between sklearn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot()),
    ])

    return ColumnTransformer(
        [("num", num, num_cols), ("cat", cat, cat_cols)],
        remainder="drop",
    )


def get_feature_names(preprocess: ColumnTransformer, num_cols, cat_cols):
    # Prefer sklearn's native feature name method when available.
    try:
        return preprocess.get_feature_names_out()
    except Exception:
        names = []
        if num_cols:
            names.extend(num_cols)
        if cat_cols:
            try:
                ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
                names.extend(ohe.get_feature_names_out(cat_cols))
            except Exception:
                names.extend(cat_cols)
        return np.array(names, dtype=object)


def collapse_feature(name: str, num_cols, cat_cols):
    base = name.split("__", 1)[1] if "__" in name else name
    if base in num_cols:
        return base
    for col in cat_cols:
        if base.startswith(col + "_"):
            return col
    return base


def train_base_model(df: pd.DataFrame, seed: int = 42):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    preprocess = build_preprocess(X_train)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=seed,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("estimator", model),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def main():
    data_path = Path("data.csv")
    if not data_path.exists():
        raise FileNotFoundError("data.csv not found")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("target column not found")

    pipeline, X_train, X_test, y_train, y_test = train_base_model(df, seed=42)
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["estimator"]

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    feature_names = get_feature_names(preprocess, num_cols, cat_cols)

    importances = estimator.feature_importances_
    order = np.argsort(importances)[::-1]

    # Aggregate importances back to original features.
    agg = {}
    for name, imp in zip(feature_names, importances):
        base = collapse_feature(name, num_cols, cat_cols)
        agg[base] = agg.get(base, 0.0) + float(imp)

    top3 = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:3]

    # Plot top 20 engineered features.
    top_n = min(20, len(order))
    top_idx = order[:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[top_idx][::-1])
    plt.yticks(range(top_n), [str(feature_names[i]) for i in top_idx[::-1]])
    plt.xlabel("importance")
    plt.title("RandomForest feature_importances_")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png")
    plt.close()

    # Stability analysis across random_state values.
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    acc_list = []
    f1_list = []
    for seed in seeds:
        pipe, _, X_te, _, y_te = train_base_model(df, seed=seed)
        pred = pipe.predict(X_te)
        acc_list.append(float(accuracy_score(y_te, pred)))
        f1_list.append(float(f1_score(y_te, pred)))

    stability = {
        "seeds": seeds,
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
    }

    # Misclassified examples (highest confidence errors).
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    wrong_mask = pred != y_test.values
    misclassified = []
    if proba is not None and wrong_mask.any():
        wrong_idx = np.where(wrong_mask)[0]
        conf = np.max(proba[wrong_mask], axis=1)
        pick = wrong_idx[np.argsort(conf)[::-1][:3]]
        for idx in pick:
            row = X_test.iloc[idx].to_dict()
            misclassified.append({
                "index": int(X_test.index[idx]),
                "true": int(y_test.iloc[idx]),
                "pred": int(pred[idx]),
                "prob": float(np.max(proba[idx])),
                "features": row,
            })

    # Simple error pattern summary on top3 features.
    pattern = []
    if wrong_mask.any():
        wrong_rows = X_test.loc[X_test.index[wrong_mask]]
        correct_rows = X_test.loc[X_test.index[~wrong_mask]]
        for feat, _ in top3:
            if feat in num_cols:
                pattern.append({
                    "feature": feat,
                    "wrong_mean": float(wrong_rows[feat].mean()),
                    "correct_mean": float(correct_rows[feat].mean()),
                })
            elif feat in cat_cols:
                wrong_mode = wrong_rows[feat].mode(dropna=True)
                correct_mode = correct_rows[feat].mode(dropna=True)
                pattern.append({
                    "feature": feat,
                    "wrong_mode": None if wrong_mode.empty else wrong_mode.iloc[0],
                    "correct_mode": None if correct_mode.empty else correct_mode.iloc[0],
                })

    # Bias/variance table (heuristic): bias=1-train_acc, variance=train_acc-test_acc.
    grid_estimators = [50, 200, 500]
    grid_depths = [None, 5, 10]
    bias_variance_rows = []
    for n_estimators in grid_estimators:
        for max_depth in grid_depths:
            X = df.drop(columns=["target"])
            y = df["target"]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            preprocess = build_preprocess(X_tr)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("estimator", model),
            ])
            pipe.fit(X_tr, y_tr)
            train_pred = pipe.predict(X_tr)
            test_pred = pipe.predict(X_te)
            train_acc = float(accuracy_score(y_tr, train_pred))
            test_acc = float(accuracy_score(y_te, test_pred))
            bias = 1.0 - train_acc
            variance = train_acc - test_acc
            bias_variance_rows.append({
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth is not None else "None",
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "bias": bias,
                "variance": variance,
            })

    pd.DataFrame(bias_variance_rows).to_csv("rf_bias_variance.csv", index=False)

    summary = {
        "top3_features": top3,
        "stability": stability,
        "misclassified_examples": misclassified,
        "error_patterns": pattern,
        "bias_variance_table": bias_variance_rows,
    }

    Path("rf_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Top3 features:", top3)
    print("Stability:", stability)
    print("Misclassified examples:", len(misclassified))
    print("Bias/variance rows:", len(bias_variance_rows))
    print("Saved: rf_feature_importance.png, rf_bias_variance.csv, rf_analysis_summary.json")


if __name__ == "__main__":
    main()

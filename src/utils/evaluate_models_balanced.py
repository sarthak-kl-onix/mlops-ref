import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

def evaluate_models_balanced(X, y, balance_method='smote', test_size=0.3, random_state=42, scoring_metric='f1'):
    """
    Evaluate multiple ML models with class imbalance handling (for churn focus).
    
    Parameters:
    - balance_method: 'smote', 'undersample', or None
    - test_size: test split ratio
    - scoring_metric: GridSearch scoring metric ('f1', 'recall', etc.)
    
    Returns:
    - results_df: DataFrame with performance of each model
    """

    # 🧩 Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 🧩 Handle imbalance if requested
    if balance_method == 'smote':
        print("🔸 Applying SMOTE oversampling for class imbalance...")
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"✅ After SMOTE: {np.bincount(y_train)}")

    elif balance_method == 'undersample':
        print("🔸 Applying RandomUnderSampler for class imbalance...")
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        print(f"✅ After Undersampling: {np.bincount(y_train)}")

    else:
        print("⚖️ No resampling applied — using class_weight balancing in models.")

    # 🧩 Define models and hyperparameters
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight='balanced', random_state=random_state),
            {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3, random_state=random_state),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]}
        ),
        "LightGBM": (
            LGBMClassifier(is_unbalance=True, random_state=random_state),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]}
        ),
        "CatBoost": (
            CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=random_state),
            {'iterations': [200], 'depth': [4, 6], 'learning_rate': [0.05, 0.1]}
        ),
    }

    results = []

    # 🧩 Train & Evaluate each model
    for name, (model, params) in models.items():
        print(f"\n🔹 Training {name}...")

        grid = GridSearchCV(model, params, scoring=scoring_metric, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Focus metrics on Churn = 0
        precision_0 = precision_score(y_test, y_pred, pos_label=0)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Balancing": balance_method,
            "Best_Params": grid.best_params_,
            "Precision_0 (Churn)": round(precision_0, 3),
            "Recall_0 (Churn)": round(recall_0, 3),
            "F1_0 (Churn)": round(f1_0, 3),
            "AUC": round(auc, 3),
            "Accuracy": round(acc, 3)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["F1_0 (Churn)", "Recall_0 (Churn)"], ascending=False).reset_index(drop=True)

    print("\n✅ Model Evaluation Summary:")
    display(results_df)

    print(f"\n🏆 Best model for Churn (Class 0): {results_df.iloc[0]['Model']} using {balance_method}")
    return results_df

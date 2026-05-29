import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_processed_data():
    print("Carregando dados processados...")
    X_train = pd.read_csv("data/processed/train_features.csv")
    y_train = pd.read_csv("data/processed/train_target.csv").squeeze()
    X_test = pd.read_csv("data/processed/test_features.csv")
    y_test = pd.read_csv("data/processed/test_target.csv").squeeze()
    print(f"Dados carregados - treino: {X_train.shape}")
    return X_train, X_test, y_train, y_test


def get_logistic_regression_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])
    return pipeline, "LogisticRegression"


def get_xgboost_pipeline():
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    return model, "XGBoost"


def train_and_evaluate_cv(model, X_train, y_train, model_name, cv_folds=5):
    print(f"Treinando {model_name}...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"{model_name} - ROC-AUC CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    model.fit(X_train, y_train)
    print(f"{model_name} treinado!")
    
    return model, scores.mean()


def save_models(lr_model, xgb_model, best_model_name, best_model):
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(lr_model, "models/logistic_regression_pipeline.pkl")
    print("Regressao Logistica salva")
    
    joblib.dump(xgb_model, "models/xgboost_pipeline.pkl")
    print("XGBoost salvo")
    
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"Melhor modelo ({best_model_name}) salvo")


def main():
    print("-Treinamento de Modelos-")
    
    X_train, X_test, y_train, y_test = load_processed_data()
    
    lr_pipeline, lr_name = get_logistic_regression_pipeline()
    lr_model, lr_score = train_and_evaluate_cv(lr_pipeline, X_train, y_train, lr_name)
    
    xgb_model, xgb_name = get_xgboost_pipeline()
    xgb_model, xgb_score = train_and_evaluate_cv(xgb_model, X_train, y_train, xgb_name)
    
    print("\nComparacao de modelos:")
    print(f"Logistic Regression - ROC-AUC: {lr_score:.4f}")
    print(f"XGBoost - ROC-AUC: {xgb_score:.4f}")

    if lr_score >= xgb_score:
        best_model_name = lr_name
        best_model = lr_model
        print(f"Melhor modelo: {best_model_name}")
    else:
        best_model_name = xgb_name
        best_model = xgb_model
        print(f"Melhor modelo: {best_model_name}")
    
    save_models(lr_model, xgb_model, best_model_name, best_model)
    
    print("Treinamento concluido!")


if __name__ == "__main__":
    main()

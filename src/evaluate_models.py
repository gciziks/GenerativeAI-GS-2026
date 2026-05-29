import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import os


def load_data_and_models():
    print("Carregando dados e modelos...")
    
    X_train = pd.read_csv("data/processed/train_features.csv")
    X_test = pd.read_csv("data/processed/test_features.csv")
    y_train = pd.read_csv("data/processed/train_target.csv").squeeze()
    y_test = pd.read_csv("data/processed/test_target.csv").squeeze()
    
    lr_model = joblib.load("models/logistic_regression_pipeline.pkl")
    xgb_model = joblib.load("models/xgboost_pipeline.pkl")
    
    return X_train, X_test, y_train, y_test, lr_model, xgb_model


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nAvaliando {model_name}...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Acuracia: {acc:.4f}")
    print(f"  Precisao: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Matriz de confusao:\n{cm}")
    
    return {
        'modelo': model_name,
        'acuracia': acc,
        'precisao': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc,
        'y_pred_proba': y_pred_proba
    }


def plot_roc_curves(results_lr, results_xgb, y_test):
    print("\nGerando curvas ROC...")
    
    plt.figure(figsize=(8, 6))
    
    for result, color, name in [
        (results_lr, 'blue', 'Logistic Regression'),
        (results_xgb, 'red', 'XGBoost')
    ]:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, color=color, label=f"{name} (AUC={result['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC - Comparacao de Modelos')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/roc_curves_comparison.png", dpi=150)
    print("Curvas ROC salvas em models/roc_curves_comparison.png")
    plt.close()


def save_results_table(results_lr, results_xgb):
    results_df = pd.DataFrame([
        {
            'Modelo': results_lr['modelo'],
            'Acuracia': results_lr['acuracia'],
            'Precisao': results_lr['precisao'],
            'Recall': results_lr['recall'],
            'F1_Score': results_lr['f1_score'],
            'ROC_AUC': results_lr['roc_auc']
        },
        {
            'Modelo': results_xgb['modelo'],
            'Acuracia': results_xgb['acuracia'],
            'Precisao': results_xgb['precisao'],
            'Recall': results_xgb['recall'],
            'F1_Score': results_xgb['f1_score'],
            'ROC_AUC': results_xgb['roc_auc']
        }
    ])
    
    results_df.to_csv("models/comparison_results.csv", index=False)
    print("\nTabela de comparacao salva em models/comparison_results.csv")
    print(results_df.to_string(index=False))


def main():
    print("- Avaliacao de Modelos -")
    
    X_train, X_test, y_train, y_test, lr_model, xgb_model = load_data_and_models()
    
    results_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    plot_roc_curves(results_lr, results_xgb, y_test)
    save_results_table(results_lr, results_xgb)
    
    print("\nAvaliacao concluida!")


if __name__ == "__main__":
    main()

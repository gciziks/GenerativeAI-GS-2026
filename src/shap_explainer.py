import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os


def load_model_and_data():
    print("Carregando modelo e dados...")
    
    model = joblib.load("models/best_model.pkl")
    X_test = pd.read_csv("data/processed/test_features.csv")
    
    print(f"Modelo carregado: {type(model).__name__}")
    print(f"Dados de teste: {X_test.shape}")
    
    return model, X_test


def get_explainer(model, X_background):
    model_type = type(model).__name__

    if 'XGB' in model_type or 'xgb' in model_type.lower():
        print("Usando TreeExplainer para XGBoost")
        explainer = shap.TreeExplainer(model)
    elif 'Logistic' in model_type:
        print("Usando LinearExplainer para Regressao Logistica")
        masker = shap.maskers.Independent(X_background)
        explainer = shap.LinearExplainer(model, masker)
    else:
        print("Usando KernelExplainer (generico)")
        explainer = shap.KernelExplainer(model.predict_proba, X_background[:100])
    
    return explainer


def compute_and_save_shap(explainer, X_test):
    print("Calculando valores SHAP... Isso pode demorar um pouco.")

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    print("Valores SHAP calculados!")
    
    os.makedirs("shap_outputs", exist_ok=True)
    
    np.save("shap_outputs/shap_values.npy", shap_values_class1)
    print("Valores SHAP salvos em shap_outputs/shap_values.npy")
    
    return shap_values_class1


def plot_summary(shap_values, X_test):
    print("Gerando visualizacoes de importancia...")
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.title("Importancia Global das Features (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_outputs/global_summary_plot.png", dpi=150)
    print("Plot salvo em shap_outputs/global_summary_plot.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )
    plt.title("Distribuicao dos Valores SHAP")
    plt.tight_layout()
    plt.savefig("shap_outputs/shap_distribution_plot.png", dpi=150)
    print("Distribuicao salva em shap_outputs/shap_distribution_plot.png")
    plt.close()


def save_feature_importance(shap_values, X_test):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    importance_df.to_csv("shap_outputs/feature_importance.csv", index=False)
    print("\nImportancia das features salva em shap_outputs/feature_importance.csv")
    print("\nTop 5 features mais importantes:")
    print(importance_df.head(5).to_string(index=False))


def main():
    print("-SHAP Explainability-")
    
    model, X_test = load_model_and_data()
    
    X_background = X_test.sample(min(100, len(X_test)), random_state=42)
    
    explainer = get_explainer(model, X_background)
    
    shap_values = compute_and_save_shap(explainer, X_test)
    
    plot_summary(shap_values, X_test)
    
    save_feature_importance(shap_values, X_test)
    
    print("\nSHAP concluido. Resultados salvos em shap_outputs/")


if __name__ == "__main__":
    main()

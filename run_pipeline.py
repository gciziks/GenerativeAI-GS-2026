import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import src.data_generation as data_generation
import src.feature_engineering as feature_engineering
import src.preprocessing as preprocessing
import src.train_models as train_models
import src.evaluate_models as evaluate_models
import src.shap_explainer as shap_explainer

def main():
    print("PIPELINE COMPLETA - Viabilidade de Mineracao em Asteroides")
    print("\n\n\n")

    print("\n[1/6] Geracao de Dados")
    print("\n\n\n")
    data_generation.main()

    print("\n[2/6] Feature Engineering")
    print("\n\n\n")
    feature_engineering.main()

    print("\n[3/6] Preprocessamento")
    print("\n\n\n")
    preprocessing.main()

    print("\n[4/6] Treinamento de Modelos")
    print("\n\n\n")
    train_models.main()

    print("\n[5/6] Avaliacao de Modelos")
    print("\n\n\n")
    evaluate_models.main()

    print("\n[6/6] Analise SHAP")
    print("\n\n\n")
    shap_explainer.main()

    print("\n\n\n")
    print("Pipeline concluido com sucesso!")
    print("Execute 'streamlit run app/app.py' para iniciar a aplicacao.")
    print("\n\n\n")

if __name__ == "__main__":
    main()

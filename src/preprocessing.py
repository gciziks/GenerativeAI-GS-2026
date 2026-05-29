import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_raw_data(path="data/raw/asteroid_mining_synthetic.csv"):
    print("Carregando dados brutos...")
    df = pd.read_csv(path)
    print(f"Dados carregados: {df.shape}")
    return df


def split_features_target(df, target_col='mining_viability'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def identify_column_types(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Colunas numericas: {len(numeric_cols)}")
    print(f"Colunas categoricas: {len(categorical_cols)}")
    return numeric_cols, categorical_cols


def preprocess_data(X, y, test_size=0.2, random_state=42):
    print("Pre-processando dados...")
    
    numeric_cols, categorical_cols = identify_column_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Split treino: {len(X_train)} | teste: {len(X_test)}")
    
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X_train[numeric_cols] = imputer_num.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer_num.transform(X_test[numeric_cols])
        print("Valores ausentes numericos tratados")
    
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])
        
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        
        cat_feature_names = encoder.get_feature_names_out(categorical_cols)
        
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test.index)
        
        X_train = pd.concat([X_train[numeric_cols], X_train_cat_df], axis=1)
        X_test = pd.concat([X_test[numeric_cols], X_test_cat_df], axis=1)
        print("Categorias codificadas com one-hot")
    
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/train_features.csv", index=False)
    X_test.to_csv("data/processed/test_features.csv", index=False)
    y_train.to_csv("data/processed/train_target.csv", index=False)
    y_test.to_csv("data/processed/test_target.csv", index=False)
    print("Dados processados salvos em data/processed/")
    
    return X_train, X_test, y_train, y_test


def main():
    print("-Preprocessamento-")
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print("Preprocessamento concluido!")


if __name__ == "__main__":
    main()

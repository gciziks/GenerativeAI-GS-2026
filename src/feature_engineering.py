import pandas as pd
import numpy as np


def create_viability_score_composite(df):
    df = df.copy()
    
    metal_norm = df['metal_concentration_pct'] / 100.0
    water_norm = df['water_ice_content_pct'] / 100.0
    delta_v_inv = 1 - (df['delta_v_km_s'] / df['delta_v_km_s'].max())
    trl_norm = df['extraction_tech_readiness'] / 9.0
    
    df['viability_score_composite'] = (
        0.4 * metal_norm +
        0.3 * water_norm +
        0.2 * delta_v_inv +
        0.1 * trl_norm
    )
    return df


def create_roi_estimate(df):
    df = df.copy()
    
    market_value_scaled = df['estimated_market_value_usd'] / 1e12
    metal_factor = df['metal_concentration_pct'] / 100.0
    delta_v_sq = df['delta_v_km_s'] ** 2
    distance_factor = 1 + df['distance_au']
    
    roi_raw = (market_value_scaled * metal_factor) / (delta_v_sq * distance_factor)
    df['roi_estimate_log'] = np.log1p(roi_raw)
    return df


def add_engineered_features(df):
    print("Criando features engenhadas...")
    
    df = create_viability_score_composite(df)
    df = create_roi_estimate(df)
    
    print("Features adicionadas: viability_score_composite, roi_estimate_log")
    return df


def main():
    print("-Feature Engineering-")
    
    df = pd.read_csv("data/raw/asteroid_mining_synthetic.csv")
    df = add_engineered_features(df)
    
    df.to_csv("data/raw/asteroid_mining_synthetic_engineered.csv", index=False)
    print(f"Dataset com features engenhadas salvo: {df.shape}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_asteroid_dataset(n_rows=1000):
    print(f"Gerando dataset com {n_rows} asteroides...")
    composition_types = ['C-type', 'S-type', 'M-type']
    composition_probs = [0.75, 0.17, 0.08]
    
    compositions = np.random.choice(composition_types, size=n_rows, p=composition_probs)

    asteroid_diameter_km = np.random.lognormal(mean=1.5, sigma=1.2, size=n_rows)
    asteroid_diameter_km = np.clip(asteroid_diameter_km, 0.1, 100)
    
    density_g_cm3 = np.where(
        compositions == 'C-type', 1.38,
        np.where(compositions == 'S-type', 2.71, 5.32)
    )
    volume_km3 = (4/3) * np.pi * (asteroid_diameter_km/2)**3
    asteroid_mass_kg = volume_km3 * (density_g_cm3 * 1e12)
    
    metal_concentration_pct = np.where(
        compositions == 'C-type', np.random.beta(2, 8, n_rows) * 30,
        np.where(compositions == 'S-type', np.random.beta(3, 5, n_rows) * 40,
                 np.random.beta(7, 2, n_rows) * 80)
    )
    
    water_ice_content_pct = np.where(
        compositions == 'C-type', np.random.beta(6, 3, n_rows) * 40,
        np.where(compositions == 'S-type', np.random.beta(2, 8, n_rows) * 15,
                 np.random.beta(1, 9, n_rows) * 5)
    )
    
    distance_au = np.random.gamma(shape=3, scale=0.5, size=n_rows) + 0.5
    distance_au = np.clip(distance_au, 0.5, 4.5)
    
    delta_v_km_s = 3 + 2 * distance_au + np.random.normal(0, 1, n_rows)
    delta_v_km_s = np.clip(delta_v_km_s, 2, 14)
    
    orbital_period_years = np.sqrt(distance_au**3)
    orbital_period_years += np.random.normal(0, 0.1, n_rows)
    
    spin_rate_hours = np.random.lognormal(mean=2.5, sigma=0.8, size=n_rows)
    spin_rate_hours = np.clip(spin_rate_hours, 0.5, 200)
    
    base_value_per_kg = np.where(
        compositions == 'C-type', 50,
        np.where(compositions == 'S-type', 200, 800)
    )
    metal_bonus = metal_concentration_pct / 100 * 5000
    estimated_market_value_usd = asteroid_mass_kg * (base_value_per_kg + metal_bonus) / 1e6
    estimated_market_value_usd *= np.random.lognormal(0, 0.3, n_rows)
    
    extraction_tech_readiness = np.random.randint(1, 10, size=n_rows)
    
    regulatory_risk_score = np.random.beta(2, 5, n_rows)
    
    viability_score = np.zeros(n_rows)
    
    viable_metal = metal_concentration_pct > 15
    viable_ice = (water_ice_content_pct > 20) & (delta_v_km_s < 6.0)
    viable_composite = viable_metal | viable_ice
    
    viability_score[viable_composite] += 0.6
    
    spin_penalty = spin_rate_hours < 2
    viability_score[spin_penalty] = 0
    
    distance_penalty = (distance_au > 2.5) & (extraction_tech_readiness < 7)
    viability_score[distance_penalty] *= 0.3
    
    tech_bonus = extraction_tech_readiness / 9.0
    viability_score += tech_bonus * 0.2
    
    risk_penalty = regulatory_risk_score > 0.7
    viability_score[risk_penalty] *= 0.5
    
    mining_viability = (viability_score > 0.5).astype(int)

    noise_mask = np.random.random(n_rows) < 0.05
    mining_viability[noise_mask] = 1 - mining_viability[noise_mask]
    
    data = {
        'asteroid_diameter_km': asteroid_diameter_km.round(2),
        'asteroid_mass_kg': asteroid_mass_kg.round(0),
        'composition_type': compositions,
        'metal_concentration_pct': metal_concentration_pct.round(2),
        'water_ice_content_pct': water_ice_content_pct.round(2),
        'delta_v_km_s': delta_v_km_s.round(2),
        'distance_au': distance_au.round(2),
        'orbital_period_years': orbital_period_years.round(2),
        'spin_rate_hours': spin_rate_hours.round(2),
        'estimated_market_value_usd': estimated_market_value_usd.round(0),
        'extraction_tech_readiness': extraction_tech_readiness,
        'regulatory_risk_score': regulatory_risk_score.round(3),
        'mining_viability': mining_viability
    }
    
    print(f"Distribuicao de viabilidade: {np.bincount(mining_viability)}")
    print("Dataset gerado!")
    return pd.DataFrame(data)


def main():
    print("Iniciando geracao de dados...")

    df = generate_asteroid_dataset(n_rows=1000)
    
    output_path = "data/raw/asteroid_mining_synthetic.csv"
    df.to_csv(output_path, index=False)
    print(f"Arquivo salvo em: {output_path}")
    print(f"Total de linhas: {len(df)}")
    print(f"Total de colunas: {len(df.columns)}")


if __name__ == "__main__":
    main()

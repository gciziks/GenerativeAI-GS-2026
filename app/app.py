import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="Analise de Viabilidade de Mineracao em Asteroides",
    page_icon="🚀",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    return model

@st.cache_resource
def load_shap_explainer(_model, X_sample):
    model_type = type(_model).__name__
    if 'XGB' in model_type:
        explainer = shap.TreeExplainer(_model)
    elif 'Logistic' in model_type:
        masker = shap.maskers.Independent(X_sample)
        explainer = shap.LinearExplainer(_model, masker)
    else:
        explainer = shap.KernelExplainer(_model.predict_proba, X_sample)
    return explainer

def get_input_features():
    st.sidebar.header("Parametros do Asteroide")
    diameter = st.sidebar.slider(
        "Diametro do asteroide (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )
    composition = st.sidebar.selectbox(
        "Tipo de composicao",
        options=["C-type", "S-type", "M-type"],
        help="C=argiloso, S=rochoso, M=metalico"
    )
    metal_conc = st.sidebar.slider(
        "Concentracao de metal (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.5
    )
    water_ice = st.sidebar.slider(
        "Conteudo de gelo/agua (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )
    delta_v = st.sidebar.slider(
        "Delta-V (km/s)",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.1
    )
    distance = st.sidebar.slider(
        "Distancia (UA)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )
    orbital_period = st.sidebar.slider(
        "Periodo orbital (anos)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.1
    )
    spin_rate = st.sidebar.slider(
        "Periodo de rotacao (horas)",
        min_value=0.5,
        max_value=100.0,
        value=12.0,
        step=0.5
    )
    market_value = st.sidebar.number_input(
        "Valor estimado de mercado (USD)",
        min_value=1_000_000,
        max_value=1_000_000_000_000_000,
        value=1_000_000_000_000,
        step=1_000_000,
        format="%d"
    )
    trl = st.sidebar.slider(
        "TRL da tecnologia de extracao (1-9)",
        min_value=1,
        max_value=9,
        value=5,
        step=1
    )
    risk_score = st.sidebar.slider(
        "Score de risco regulatório (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    input_data = {
        'asteroid_diameter_km': diameter,
        'asteroid_mass_kg': diameter ** 3 * 1000,
        'composition_type': composition,
        'metal_concentration_pct': metal_conc,
        'water_ice_content_pct': water_ice,
        'delta_v_km_s': delta_v,
        'distance_au': distance,
        'orbital_period_years': orbital_period,
        'spin_rate_hours': spin_rate,
        'estimated_market_value_usd': market_value,
        'extraction_tech_readiness': trl,
        'regulatory_risk_score': risk_score
    }
    
    return pd.DataFrame([input_data])

def preprocess_input(df_input):
    composition_encoded = pd.get_dummies(df_input['composition_type'], prefix='composition_type')
    df_processed = pd.concat([df_input.drop('composition_type', axis=1), composition_encoded], axis=1)
    
    expected_cols = [
        'asteroid_diameter_km', 'asteroid_mass_kg', 'metal_concentration_pct',
        'water_ice_content_pct', 'delta_v_km_s', 'distance_au', 'orbital_period_years',
        'spin_rate_hours', 'estimated_market_value_usd', 'extraction_tech_readiness',
        'regulatory_risk_score', 'composition_type_C-type', 'composition_type_M-type',
        'composition_type_S-type'
    ]
    
    for col in expected_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    df_processed = df_processed[expected_cols]
    
    return df_processed

FEATURE_EXPLANATIONS = {
    'asteroid_diameter_km': (
        "Diametro do asteroide",
        "Asteroides maiores contem maior volume de minerais e metais, tornando a missao mais rentavel. "
        "No contexto de mineracao espacial, um diametro maior significa mais recursos disponiveis por missao, "
        "diluindo os custos fixos de lancamento e operacao."
    ),
    'asteroid_mass_kg': (
        "Massa do asteroide",
        "A massa determina diretamente a quantidade total de material exploravel. "
        "Quanto maior a massa, maior o potencial de retorno economico — especialmente para metais de alto valor como platina e iridio."
    ),
    'metal_concentration_pct': (
        "Concentracao de metal (%)",
        "Metais preciosos e industriais (platina, niquel, ferro) sao o principal alvo economico da mineracao de asteroides. "
        "Uma alta concentracao significa maior rendimento por tonelada extraida, impactando diretamente a lucratividade da missao."
    ),
    'water_ice_content_pct': (
        "Conteudo de gelo/agua (%)",
        "A agua no espaco tem valor duplo: pode ser separada em hidrogenio e oxigenio para propelente de foguetes, "
        "ou usada para suporte a vida em missoes tripuladas. Isso reduz a necessidade de transportar suprimentos da Terra, "
        "aumentando significativamente a viabilidade logistica e economica da missao."
    ),
    'delta_v_km_s': (
        "Delta-V (km/s)",
        "Delta-V representa o esforco de propulsao necessario para alcancar o asteroide. "
        "Valores mais altos exigem mais combustivel, aumentando drasticamente o custo e a complexidade da missao. "
        "E um dos principais limitantes fisicos da mineracao espacial."
    ),
    'distance_au': (
        "Distancia (UA)",
        "Quanto mais distante o asteroide, maior o tempo de viagem, o custo de comunicacao e o consumo de propelente. "
        "Asteroides proximos a Terra (NEOs) sao os candidatos mais viáveis economicamente."
    ),
    'orbital_period_years': (
        "Periodo orbital (anos)",
        "O periodo orbital define com que frequencia o asteroide se aproxima da Terra em posicao favoravel para uma missao. "
        "Periodos muito longos reduzem as janelas de lancamento disponiveis, dificultando o planejamento operacional."
    ),
    'spin_rate_hours': (
        "Periodo de rotacao (horas)",
        "Asteroides com rotacao muito rapida sao extremamente dificeis de acessar e minerar com seguranca — "
        "ancora-los ou pousar neles requer tecnologias avancadas. Uma rotacao lenta facilita as operacoes de extracao."
    ),
    'estimated_market_value_usd': (
        "Valor estimado de mercado (USD)",
        "Representa o valor total dos recursos minerais presentes, estimado com base na composicao e massa do asteroide. "
        "E o indicador mais direto do potencial de retorno financeiro da missao."
    ),
    'extraction_tech_readiness': (
        "TRL da tecnologia de extracao",
        "O Technology Readiness Level (TRL) mede a maturidade da tecnologia de extracao disponivel. "
        "Tecnologias com TRL baixo ainda estao em fase experimental, elevando o risco tecnico e o custo de desenvolvimento. "
        "TRL alto indica tecnologia pronta para uso operacional."
    ),
    'regulatory_risk_score': (
        "Score de risco regulatorio",
        "O ambiente legal para mineracao de asteroides ainda esta em desenvolvimento. "
        "Um score alto indica maior incerteza juridica — sobre direitos de propriedade, tratados internacionais e licencas — "
        "o que pode inviabilizar ou atrasar uma missao mesmo que tecnicamente viável."
    ),
    'composition_type_C-type': (
        "Composicao: Tipo C (argiloso)",
        "Asteroides tipo C sao ricos em carbono, agua e minerais argilosos. "
        "Embora menos ricos em metais preciosos, seu alto conteudo de agua os torna valiosos para producao de propelente in-situ."
    ),
    'composition_type_M-type': (
        "Composicao: Tipo M (metalico)",
        "Asteroides tipo M sao compostos principalmente de ferro e niquel, com tracos de platina e outros metais preciosos. "
        "Sao os alvos mais lucrativos para mineracao de metais industriais e preciosos."
    ),
    'composition_type_S-type': (
        "Composicao: Tipo S (rochoso)",
        "Asteroides tipo S contem silicatos, magnesio e ferro em menor quantidade. "
        "Tem valor economico intermediario — menos agua que tipo C e menos metais que tipo M."
    ),
}

def main():
    st.title("Analise de Viabilidade de Mineracao em Asteroides")
    st.markdown("---")
    
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Modelo nao encontrado. Execute o treinamento primeiro!")
        return
    
    df_input = get_input_features()
    df_processed = preprocess_input(df_input)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Fazer Previsao", type="primary"):
        st.markdown("### Resultado da Analise")
        
        with st.spinner("Calculando probabilidade..."):
            proba = model.predict_proba(df_processed)[0]
            prediction = model.predict(df_processed)[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success(" **VIAVEL** para mineracao")
            else:
                st.error(" **NAO VIAVEL** para mineracao")
        
        with col2:
            st.metric(
                label="Probabilidade de viabilidade",
                value=f"{proba[1]:.1%}"
            )
        
        with col3:
            confidence = max(proba)
            st.metric(
                label="Confianca da previsao",
                value=f"{confidence:.1%}"
            )
        
        st.markdown("---")
        st.subheader("Explicacao da Previsao (SHAP)")
        
        with st.spinner("Calculando explicacao..."):
            try:
                X_sample = pd.read_csv("data/processed/test_features.csv")
                if len(X_sample) > 100:
                    X_sample = X_sample.sample(100, random_state=42)
                
                explainer = load_shap_explainer(model, X_sample)
                shap_values = explainer.shap_values(df_processed)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                st.write("Top fatores que influenciaram esta previsao:")
                
                feature_importance = list(zip(df_processed.columns, shap_values[0]))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feature, value in feature_importance[:5]:
                    direction = "aumenta" if value > 0 else "diminui"
                    magnitude = abs(value)
                    if magnitude > 0.1:
                        impact = "muito"
                    elif magnitude > 0.05:
                        impact = "moderadamente"
                    else:
                        impact = "pouco"
                    
                    label, explanation = FEATURE_EXPLANATIONS.get(
                        feature, (feature, "Sem explicacao disponivel.")
                    )
                    with st.expander(f"{'🟢' if value > 0 else '🔴'} **{label}** — {direction} a viabilidade ({impact})"):
                        st.markdown(f"**Por que este fator importa?**\n\n{explanation}")
                        st.caption(f"Impacto SHAP: `{value:+.4f}` | Direcao: {direction} | Magnitude: {impact}")
                
            except Exception as e:
                st.warning(f"Nao foi possivel gerar explicacao SHAP: {e}")
        
        st.markdown("---")
        st.subheader("Dados do Asteroide")
        st.dataframe(df_input.T, use_container_width=True)
        
        csv = df_input.to_csv(index=False)
        st.download_button(
            label="Download dos dados como CSV",
            data=csv,
            file_name="asteroide_analise.csv",
            mime="text/csv"
        )
if __name__ == "__main__":
    main()

# Global Solution - Generative AI
###  Classificador de Viabilidade de Mineração em Asteroides

## Grupo
- Gabriel Ciziks - RM98215
- Victor Nuzzi - RM98209
- Lucca Sabatini - RM98169
- Cassio Valezzi - RM551059

---

## Visão Geral do Projeto
- **Tipo de Problema**: Classificação Binária
- **Modelos**: Regressão Logística vs. XGBoost
- **Deploy**: Aplicação Streamlit interativa
- **Link do Projeto para Testar**: 
https://generativeai-gs-2026-tlnwavqc7kdgimu3dmnpqp.streamlit.app/

Este projeto implementa uma pipeline completa de Machine Learning para prever se um asteroide é viável para operações de mineração espacial, baseado em características orbitais, composição química, parâmetros econômicos e tecnológicos.

---

## Contexto do Problema

### Mineração de Asteroides: O Próximo Frontier Econômico

A mineração de asteroides representa uma das fronteiras mais promissoras e desafiadoras da exploração espacial moderna. Com a crescente demanda por recursos terrestres escassos e o avanço das tecnologias espaciais, a exploração de corpos celestes próximos à Terra tornou-se economicamente viável e estrategicamente relevante.

### O Desafio

**Problema de Negócio:** Determinar quais asteroides são economicamente viáveis para operações de mineração, considerando múltiplos fatores técnicos, econômicos e regulatórios.

**Complexidades envolvidas:**
- **Custo de acesso**: O delta-v (variação de velocidade necessária) determina o custo de chegar ao asteroide
- **Composição variável**: Diferentes tipos de asteroides (C, S, M) têm composições químicas distintas
- **Valor econômico**: Metais preciosos (platina, irídio) vs. recursos voláteis (água, propelentes)
- **Riscos tecnológicos**: Maturidade das tecnologias de extração (TRL)
- **Incertezas regulatórias**: Ambiente legal ainda em desenvolvimento para propriedade espacial

**Impacto:** Uma classificação precisa permite empresas espaciais e agências governamentais priorizar alvos de missão, otimizando investimentos bilionários em exploração espacial.

---

## Dataset

- **Fonte**: Geração sintética via `src/data_generation.py`
- **Tamanho**: 1.000+ linhas, 12+ features
- **Target**: `mining_viability` (0 = Não Viável, 1 = Viável)
- **Features**: Diâmetro, massa, tipo de composição, concentração de metais, conteúdo de gelo/água, delta-v, distância, período orbital, rotação, valor de mercado, TRL da tecnologia, risco regulatório

## Arquitetura do Pipeline

1. **Geração de Dados**: Dataset sintético com correlações realistas do domínio espacial
2. **Preprocessamento**: Input, encoding, escalonamento (sem vazamento de dados)
3. **Feature Engineering**: 2 features específicas do domínio (viability_score_composite, roi_estimate_log)
4. **Treinamento**: Validação cruzada estratificada (5 folds)
5. **Avaliação**: Acurácia, Precisão, Recall, F1-Score, ROC-AUC
6. **Explicabilidade**: SHAP (importância global e explicações individuais)
7. **Deploy**: Interface Streamlit para previsões em tempo real

## Estrutura do Repositório

```
GenerativeAI-GS-2026/
├── data/
│   ├── raw/                    # Dataset original e com features engenhadas
│   └── processed/              # Dados preprocessados (train/test splits)
├── src/
│   ├── data_generation.py      # Script de geração de dados (esqueleto)
│   ├── preprocessing.py        # Pipeline de preprocessamento
│   ├── feature_engineering.py  # Criação de features
│   ├── train_models.py         # Treinamento de modelos
│   ├── evaluate_models.py      # Avaliação e comparação
│   └── shap_explainer.py       # Análise de explicabilidade
├── models/                     # Modelos serializados (.pkl)
├── shap_outputs/               # Visualizações e dados SHAP
├── app/
│   └── app.py                  # Aplicação Streamlit
├── notebooks/                  # Notebooks de análise exploratória
├── run_pipeline.py         # Executa o pipeline completo em um comando
├── requirements.txt
└── README.md
```

## Execução

### Opção 1: Pipeline completo

```bash
# 1. Instalar as dependências
pip install -r requirements.txt

# 2. Executar todo o pipeline de uma vez
python run_pipeline.py

# 3. Iniciar aplicação Streamlit
streamlit run app/app.py
```

### Opção 2: Passo a passo

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Gerar dados sintéticos
python src/data_generation.py

# 3. Opcional: adicionar features engenhadas
python src/feature_engineering.py

# 4. Preprocessar dados
python src/preprocessing.py

# 5. Treinar modelos
python src/train_models.py

# 6. Avaliar modelos
python src/evaluate_models.py

# 7. Gerar análise SHAP
python src/shap_explainer.py

# 8. Iniciar aplicação Streamlit
streamlit run app/app.py
```


## Principais Resultados

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|--------|----------|----------|--------|----------|---------|
| Regressão Logística | 82.5% | 72.1% | 84.9% | 0.780 | 0.892 |
| XGBoost | 92.0% | 88.0% | 90.4% | 0.892 | 0.927 |

**Melhor Modelo**: XGBoost (ROC-AUC 0.927)

## Modelos Testados

### 1. Regressão Logística (Baseline)
**Tipo:** Modelo linear de classificação
**Características:**
- Regularização L2 (Ridge) para evitar overfitting
- Balanceamento de classes via `class_weight='balanced'`
- StandardScaler para normalização de features

**Hiperparâmetros:**
```python
C=1.0                    # Regularização
solver='lbfgs'           # Otimizador
max_iter=1000            # Iterações máximas
class_weight='balanced'  # Balanceamento
```

**Justificativa:** Serve como baseline interpretável, permitindo entender os coeficientes lineares de cada feature.

### 2. XGBoost (Gradient Boosting)
**Tipo:** Ensemble de árvores de decisão com gradient boosting
**Características:**
- Alta capacidade de capturar relações não-lineares
- Robustez a outliers e dados faltantes
- Otimização automática de ganho de informação

**Hiperparâmetros:**
```python
n_estimators=200         # Número de árvores
max_depth=6              # Profundidade máxima
learning_rate=0.1        # Taxa de aprendizado
subsample=0.8            # Amostragem por árvore
colsample_bytree=0.8     # Amostragem de features
eval_metric='logloss'    # Métrica de otimização
```

**Justificativa:** Estado da arte para problemas tabulares, frequentemente superior em datasets com features heterogêneas.

### Critério de Seleção do Melhor Modelo
- **Métrica principal:** ROC-AUC (área sob a curva ROC)
- **Validação:** Cross-validation estratificado com 5 folds
- O modelo com maior ROC-AUC médio na validação cruzada é selecionado

---

## Interpretação com SHAP (SHapley Additive exPlanations)

### O que é SHAP?

SHAP é uma técnica de explicabilidade baseada na teoria dos jogos que atribui a cada feature uma importância para cada predição, garantindo propriedades matemáticas desejáveis como consistência e local accuracy.

### Resultados da Análise SHAP

As 5 features mais importantes para determinar viabilidade de mineração:

| Rank | Feature | SHAP Value | Interpretação |
|------|---------|------------|---------------|
| 1 | `delta_v_km_s` | 2.48 | Custo de acesso orbital; **fator mais determinante** - baixo delta-v aumenta drasticamente a viabilidade |
| 2 | `metal_concentration_pct` | 1.65 | Concentração metálica; alta concentração aumenta fortemente a probabilidade |
| 3 | `water_ice_content_pct` | 0.78 | Presença de gelo/água; relevante para asteroides tipo C próximos |
| 4 | `orbital_period_years` | 0.37 | Período orbital; períodos curtos favorecem janelas de acesso frequentes |
| 5 | `distance_au` | 0.36 | Distância; maior distância eleva custo e reduz viabilidade |

### Visualizações Geradas

1. **Global Summary Plot** (`shap_outputs/global_summary_plot.png`)
   - Ranking das features por importância média
   - Barras horizontais com valores SHAP absolutos médios

2. **Distribution Plot** (`shap_outputs/shap_distribution_plot.png`)
   - Distribuição dos valores SHAP para cada feature
   - Mostra como cada feature impacta positiva/negativamente

3. **Feature Importance CSV** (`shap_outputs/feature_importance.csv`)
   - Tabela detalhada com valores SHAP para análise quantitativa

### Insights de Negócio

**Fatores Críticos para Viabilidade:**
- **Delta-V baixo (< 6 km/s):** Asteroides próximos da Terra com baixo custo de acesso são prioritários
- **Alto teor metálico (> 15%):** Asteroides tipo M e S ricos em metais são alvos lucrativos
- **Presença de água (> 20%):** Importante para asteroides tipo C usados como "postos de gasolina" espaciais

**Fatores de Risco:**
- **Rotação rápida (< 2h):** Dificulta operações de atraque e mineração
- **Distância grande (> 2.5 UA) + TRL baixo:** Combinação de alto custo e tecnologia imatura é inviável

---

### Funcionalidades da Aplicação

- **Input interativo**: Ajuste os parâmetros do asteroide via sliders intuitivos
- **Previsão em tempo real**: Obtenha a classificação de viabilidade instantaneamente
- **Explicação SHAP**: Entenda quais fatores mais influenciaram a decisão
- **Download de dados**: Exporte os dados do asteroide analisado em CSV

### Executar localmente

```bash
streamlit run app/app.py
```

Acesse em: `http://localhost:8501`

---

## Metodologia Detalhada

### 1. Engenharia de Dados

**Geração Sintética com Correlações Realistas:**
- Dados gerados via simulação estatística baseada em literatura astronômica
- Correlações entre composição química, distância orbital e viabilidade
- Adição controlada de ruído (5%) para simular incertezas reais

### 2. Feature Engineering

**Features Criadas:**
- `viability_score_composite`: Score ponderado combinando metal, gelo, delta-v e TRL
  - Formula: `0.4*metal + 0.3*gelo + 0.2*(1-delta_v) + 0.1*TRL`
- `roi_estimate_log`: Log do retorno sobre investimento estimado
  - Considera valor de mercado, concentração metálica, delta-v e distância

### 3. Preprocessamento

**Pipeline robusto sem vazamento de dados:**
- One-hot encoding para variáveis categóricas
- Imputação de valores ausentes (mediana para numéricas, moda para categóricas)
- Split estratificado (80/20) mantendo proporção da classe alvo

### 4. Treinamento e Validação

**Estratégia de validação:**
- Cross-validation estratificado com 5 folds
- Otimização de hiperparâmetros via validação cruzada
- Balanceamento de classes via `class_weight` e `scale_pos_weight`

### 5. Avaliação Comparativa

**Métricas utilizadas:**
- **ROC-AUC**: Principal métrica (área sob a curva ROC)
- **F1-Score**: Equilíbrio entre precisão e recall
- **Precision/Recall**: Análise detalhada de erros tipo I e II
- **Matriz de Confusão**: Visualização de acertos e erros

### 6. Explicabilidade com SHAP

**Análise de importância global e local:**
- Identificação das features mais impactantes para o modelo
- Explicações individuais para cada previsão
- Visualizações: summary plots, distribution plots, feature importance ranking
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Teste ML",
    layout="wide"
)

# --- 2. CARREGAMENTO DO MODELO DE ML E FEATURES ---
@st.cache_resource
def load_ml_model_and_features():
    """
    Carrega o modelo de Machine Learning treinado e a lista de features.
    """
    try:
        with open('modelo_classificacao_violencia.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('features_modelo.pkl', 'rb') as features_file:
            features = pickle.load(features_file)
        return model, features
    except FileNotFoundError:
        st.error("Erro: Arquivos do modelo de ML (modelo_classificacao_violencia.pkl ou features_modelo.pkl) não encontrados.")
        st.info("Por favor, execute o script de treinamento do modelo ('treinar_modelo.py') primeiro para gerá-los.")
        st.stop()

# Carrega o modelo e as features ao iniciar o aplicativo
ml_model, ml_model_features = load_ml_model_and_features()

# --- 3. BARRA LATERAL (SIDEBAR) APENAS COM A SEÇÃO DE ML ---
with st.sidebar:
    st.title("Teste de Funcionalidade ML")
    st.markdown("---")

    st.header("Previsão de Risco de Agressão (ML)")
    st.markdown("Insira as características para estimar o risco de violência:")

    # Mapeamentos (simplificados para este teste, mas devem ser consistentes com o treinamento)
    education_map_for_model = {'Desconhecido': -1, 'Sem Escolaridade': 0, 'Primário': 1, 'Ensino Médio': 2, 'Superior': 3}
    employment_map_for_model = {'Desempregada': 0, 'Parc. Empregada': 1, 'Empregada': 2, 'Desconhecido': -1}
    marital_status_map_for_model = {'Solteira': 0, 'Casada': 1, 'Desconhecido': -1}

    # Campos de entrada para o modelo
    age_ml = st.slider("Idade da Pessoa", 18, 90, 30, key='age_ml_test')
    education_input_ml = st.selectbox("Nível de Escolaridade", options=list(education_map_for_model.keys()), key='edu_ml_test')
    employment_input_ml = st.selectbox("Situação de Emprego", options=list(employment_map_for_model.keys()), key='emp_ml_test')
    income_ml = st.number_input("Renda Anual (aproximada, em reais)", min_value=0, value=15000, step=1000, key='income_ml_test')
    marital_status_input_ml = st.selectbox("Estado Civil", options=list(marital_status_map_for_model.keys()), key='marital_ml_test')

    st.markdown("---")
    if st.button("Analisar Risco de Agressão", key='predict_button_test'):
        education_val = education_map_for_model[education_input_ml]
        employment_val = employment_map_for_model[employment_input_ml]
        marital_status_val = marital_status_map_for_model[marital_status_input_ml]

        input_data_dict = {
            'age': age_ml,
            'education': education_val,
            'employment': employment_val,
            'income': income_ml,
            'marital_status': marital_status_val
        }

        input_df = pd.DataFrame([input_data_dict], columns=ml_model_features)

        prediction_proba = ml_model.predict_proba(input_df)
        chance_of_violence = prediction_proba[0][1] * 100

        st.session_state['ml_prediction_result_test'] = {
            'chance': chance_of_violence,
            'age': age_ml,
            'education': education_input_ml,
            'employment': employment_input_ml,
            'income': income_ml,
            'marital_status': marital_status_input_ml
        }
        st.rerun()

# --- 4. CONTEÚDO PRINCIPAL DO DASHBOARD (APENAS RESULTADO ML) ---
st.title("Resultado do Teste de ML")

if 'ml_prediction_result_test' in st.session_state:
    st.divider()
    st.header("Resultado da Previsão de Risco de Agressão")
    result = st.session_state['ml_prediction_result_test']
    chance = result['chance']

    st.write(f"Para uma pessoa com as seguintes características:")
    st.markdown(f"- **Idade:** {result['age']}")
    st.markdown(f"- **Escolaridade:** {result['education']}")
    st.markdown(f"- **Situação Profissional:** {result['employment']}")
    st.markdown(f"- **Renda Anual:** R$ {result['income']:,}".replace(",", "."))
    st.markdown(f"- **Estado Civil:** {result['marital_status']}")

    if chance >= 70:
        st.error(f"⚠️ **ALTO RISCO!** Esta pessoa tem **{chance:.2f}%** de chance de sofrer agressão.")
        st.write("É recomendado atenção imediata e suporte para este caso.")
    elif chance >= 40:
        st.warning(f"🟡 **MÉDIO RISCO.** Esta pessoa tem **{chance:.2f}%** de chance de sofrer agressão.")
        st.write("Monitoramento e ações preventivas podem ser importantes.")
    else:
        st.success(f"✅ **BAIXO RISCO.** Esta pessoa tem **{chance:.2f}%** de chance de sofrer agressão.")
        st.write("Ainda assim, a prevenção e a conscientização são sempre fundamentais.")

    st.markdown("---")
    st.write("*(Esta previsão é baseada em um modelo de Machine Learning e deve ser usada como um indicativo, não como um diagnóstico definitivo.)*")
    del st.session_state['ml_prediction_result_test']

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np # Necessário para manipulação de dados, se houver NaNs etc.

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Violência Doméstica",
    layout="wide"
)

# --- Mapeamentos Globais ---
# Estes dicionários precisam estar no escopo global para serem acessíveis
# tanto pela função carregar_dados quanto pela seção de ML.
niveis_educacao_map = {-1: 'Desconhecido', 0: 'Sem Escolaridade', 1: 'Primário', 2: 'Ensino Médio', 3: 'Superior'}
marital_status_map = {0: 'Solteira', 1: 'Casada', -1: 'Desconhecido'}
employment_map = {0: 'Desempregada', 1: 'Parc. Empregada', 2: 'Empregada', -1: 'Desconhecido'}


# --- 2. CARREGAMENTO DO MODELO DE ML E FEATURES ---
# Usamos st.cache_resource para carregar o modelo apenas uma vez
@st.cache_resource
def load_ml_model_and_features():
    """
    Carrega o modelo de Machine Learning treinado e a lista de features
    necessárias para a previsão.
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
        st.stop() # Para a execução do Streamlit se os arquivos não existirem

# Carrega o modelo e as features ao iniciar o aplicativo
ml_model, ml_model_features = load_ml_model_and_features()

# --- 3. CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS PARA O DASHBOARD ---
@st.cache_data
def carregar_dados():
    """
    Carrega o dataset 'dados_tratados.csv' e aplica mapeamentos para rótulos legíveis.
    """
    df = pd.read_csv('dados_tratados.csv')

    # Os mapeamentos agora usam os dicionários globais
    df['education_label'] = df['education'].map(niveis_educacao_map)
    df['marital_status_label'] = df['marital_status'].map(marital_status_map)
    df['employment_label'] = df['employment'].map(employment_map)

    df['violence_label'] = df['violence'].map({True: 'Sim', False: 'Não', None: 'Desconhecido'}) # Adiciona None para 'Desconhecido'

    df['income_category'] = pd.cut(df['income'], bins=[-1, 1000, 2000, 3000, float('inf')], labels=["Até 1k", "1k-2k", "2k-3k", "3k+"])
    return df

# Executa o carregamento dos dados para o dashboard
df_original = carregar_dados()

# --- 4. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("Dashboard")

    with st.expander("Informações do Projeto"):
        st.markdown("""
        **Disciplina:** Ciências de Dados 35T34
        * **Docente:** Luiz Affonso Henderson Guedes de Oliveira
        * **Discentes:** Ramylla Luiza Barbalho Gomes Bezerra, Juan Pablo Hortencio Ferreira
        """)

    st.markdown("---") # Separador visual

    # Filtros de Análise dentro de um expander
    with st.expander("Filtros de Análise", expanded=True): # Deixando expandido por padrão
        # Filtros sociodemográficos e socioeconômicos agora diretos na sidebar
        estado_civil_selecionado = st.multiselect("Estado Civil:", options=df_original['marital_status_label'].unique(), default=df_original['marital_status_label'].unique())
        idade_min, idade_max = int(df_original['age'].min()), int(df_original['age'].max())
        faixa_idade = st.slider("Faixa de Idade:", min_value=idade_min, max_value=idade_max, value=(idade_min, idade_max))
        educacao_selecionada = st.multiselect("Nível de Escolaridade:", options=df_original['education_label'].unique(), default=df_original['education_label'].unique())
        situacao_selecionada = st.multiselect("Situação Profissional:", options=df_original['employment_label'].unique(), default=df_original['employment_label'].unique())

    # Aplicação dos filtros ao DataFrame (fora do expander, pois df_filtrado é usado em todo o dashboard)
    df_filtrado = df_original[
        (df_original['education_label'].isin(educacao_selecionada)) &
        (df_original['age'].between(faixa_idade[0], faixa_idade[1])) &
        (df_original['employment_label'].isin(situacao_selecionada)) &
        (df_original['marital_status_label'].isin(estado_civil_selecionado))
    ]

    st.markdown("---") # Separador visual

    # --- SEÇÃO DE PREVISÃO DE ML NA SIDEBAR ---
    # Colocando a seção de ML dentro de um expander
    with st.expander("Previsão de Risco de Agressão", expanded=False): # Deixando recolhido por padrão
        # Descrição da funcionalidade de ML para aparecer no dashboard
        # Ajustado para usar st.markdown com quebras de linha para melhor visualização
        st.markdown("""
        * **Tipo de ML:** Classificação
        * **Objetivo:** Prever o risco de uma pessoa sofrer agressão (binário: sim/não)
        * **Modelo utilizado:** RandomForestClassifier
        """)

        st.markdown("Insira as características para estimar o risco de violência:")

        # Mapeamentos para converter de volta para os valores numéricos esperados pelo modelo
        # Estes devem ser consistentes com o script de treinamento
        education_map_for_model = {v: k for k, v in niveis_educacao_map.items()}
        employment_map_for_model = {v: k for k, v in employment_map.items()}
        marital_status_map_for_model = {v: k for k, v in marital_status_map.items()}

        # Campos de entrada para o modelo
        age_ml = st.slider("Idade da Pessoa", 18, 90, 30, key='age_ml')
        education_input_ml = st.selectbox("Nível de Escolaridade", options=list(education_map_for_model.keys()), key='edu_ml')
        employment_input_ml = st.selectbox("Situação de Emprego", options=list(employment_map_for_model.keys()), key='emp_ml')
        income_ml = st.number_input("Renda Anual (aproximada, em reais)", min_value=0, value=15000, step=1000, key='income_ml')
        marital_status_input_ml = st.selectbox("Estado Civil", options=list(marital_status_map_for_model.keys()), key='marital_ml')

        st.markdown("---")
        if st.button("Analisar Risco de Agressão"):
            # Converter as entradas do usuário de volta para o formato numérico do modelo
            education_val = education_map_for_model[education_input_ml]
            employment_val = employment_map_for_model[employment_input_ml]
            marital_status_val = marital_status_map_for_model[marital_status_input_ml]

            # Criar um DataFrame com as entradas do usuário
            # As colunas devem estar na mesma ordem que o modelo foi treinado!
            input_data_dict = {
                'age': age_ml,
                'education': education_val,
                'employment': employment_val,
                'income': income_ml,
                'marital_status': marital_status_val
            }

            # Cria um DataFrame com uma única linha, garantindo a ordem das colunas
            # Usamos ml_model_features para garantir que a ordem das colunas seja a mesma do treinamento
            input_df = pd.DataFrame([input_data_dict], columns=ml_model_features)

            # Fazer a Previsão de Probabilidade
            # predict_proba retorna as probabilidades para cada classe [prob_classe_0, prob_classe_1]
            # Onde classe_0 é 'Não Violência' (0) e classe_1 é 'Violência' (1)
            prediction_proba = ml_model.predict_proba(input_df)
            chance_of_violence = prediction_proba[0][1] * 100 # Probabilidade da classe 'Violência' (índice 1)

            # Armazenar o resultado na sessão para exibição no corpo principal
            st.session_state['ml_prediction_result'] = {
                'chance': chance_of_violence,
                'age': age_ml,
                'education': education_input_ml,
                'employment': employment_input_ml,
                'income': income_ml,
                'marital_status': marital_status_input_ml
            }
            st.rerun() # Força o re-render para exibir o resultado no corpo principal

# --- 5. CONTEÚDO PRINCIPAL DO DASHBOARD ---
st.header("Análise Interativa de Violência Doméstica")
st.caption(f"Violência doméstica contra mulheres em uma área rural específica de um país em desenvolvimento.")
st.caption(f"Última atualização: 17 de junho de 2025")
st.divider()

# --- CARD DE MÉTRICAS PRINCIPAIS ---
with st.container(border=True):
    st.subheader("Resumo dos dados")
    if not df_filtrado.empty:
        total_casos = df_filtrado.shape[0]
        # Filtra para contar apenas casos onde 'violence' é True (Sim) e não None
        casos_violencia = df_filtrado['violence'].sum() # True é 1, False é 0
        taxa_violencia = (casos_violencia / total_casos) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Respondentes", f"{total_casos}")
        col2.metric("Casos de Violência Reportados", f"{casos_violencia}")
        col3.metric("Taxa de Violência", f"{taxa_violencia:.2f}%")
    else:
        st.warning("Nenhum dado corresponde aos filtros selecionados.")

# --- SEÇÃO DE RESULTADO DA PREVISÃO DE ML ---
if 'ml_prediction_result' in st.session_state:
    st.divider()
    st.header("Resultado da Previsão de Risco de Agressão")
    result = st.session_state['ml_prediction_result']
    chance = result['chance']

    st.write(f"Para uma pessoa com as seguintes características:")
    st.markdown(f"- **Idade:** {result['age']}")
    st.markdown(f"- **Escolaridade:** {result['education']}")
    st.markdown(f"- **Situação Profissional:** {result['employment']}")
    st.markdown(f"- **Renda Anual:** R$ {result['income']:,}".replace(",", ".")) # Formatação para reais
    st.markdown(f"- **Estado Civil:** {result['marital_status']}")

    if chance >= 70:
        st.error(f"⚠️ **ALTO RISCO!** Esta pessoa tem **{chance:.2f}%** de chance de sofrer agressão.")
        st.write("É recomendado atenção imediata e suporte para este caso.")
    elif chance >= 40:
        st.warning(f"🟡 **MÉDIO RISCO.** Esta pessoa tem **{chance:.2f}%** de chance de sofrer agressão.")
        st.write("Monitoramento e ações preventivas podem ser importantes.")
    else:
        st.success(f"✅ **BAIXO RISCO.** Esta pessoa tem **{chance:.2f}%** de chance de sofrer agressão.")

    st.markdown("---")
    del st.session_state['ml_prediction_result'] # Limpa o estado após exibir

# --- SEÇÃO DE ANÁLISE GRÁFICA ---
st.header("Análise Gráfica Detalhada")

PALETA_AZUL_VERMELHO = {'Sim': '#DC143C', 'Não': '#4682B4', 'Desconhecido': '#A9A9A9'} # Adicionado 'Desconhecido'

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("###### Incidência por Estado Civil")
    fig1 = px.histogram(df_filtrado, x='marital_status_label', color='violence_label', barmode='group', color_discrete_map=PALETA_AZUL_VERMELHO)
    fig1.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None, template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("###### Incidência por Faixa de Renda")
    fig2 = px.histogram(df_filtrado, x='income_category', color='violence_label', barmode='group', category_orders={"income_category": ["Até 1k", "1k-2k", "2k-3k", "3k+"]}, color_discrete_map=PALETA_AZUL_VERMELHO)
    fig2.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown("###### Incidência por Situação Profissional")
    ordem_emprego = ['Desempregada', 'Parc. Empregada', 'Empregada', 'Desconhecido']
    fig4 = px.histogram(df_filtrado, x='employment_label', color='violence_label', barmode='group', color_discrete_map=PALETA_AZUL_VERMELHO, category_orders={"employment_label": ordem_emprego})
    fig4.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None, template='plotly_white')
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

with st.container(border=True):
    st.subheader("Distribuição de Idades na Amostra")
    fig3 = px.histogram(df_filtrado, x='age', nbins=20, color_discrete_sequence=['#1f77b4'])
    fig3.update_traces(marker_line_color='black', marker_line_width=1)
    fig3.update_layout(height=400, template='plotly_white', bargap=0.1)
    st.plotly_chart(fig3, use_container_width=True)

with st.container(border=True):
    st.subheader("Matriz de Correlação entre Variáveis")
    correlation_matrix = df_original.drop(columns=['id']).corr(numeric_only=True)
    fig5 = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
    fig5.update_layout(height=500, title_text='Correlação de Pearson', title_x=0.5, template='plotly_white')
    st.plotly_chart(fig5, use_container_width=True)

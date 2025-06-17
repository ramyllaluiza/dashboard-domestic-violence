#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ciências de Dados 35T34",
    layout="wide"
)

# Define um tema visual limpo para os gráficos do Plotly
px.defaults.template = "plotly_white"

# --- 2. CARREGAMENTO E PROCESSAMENTO DE DADOS ---

@st.cache_data
def carregar_dados():
    """
    Carrega e pré-processa o conjunto de dados.
    As transformações são executadas apenas uma vez para otimizar o desempenho.
    """
    df = pd.read_csv('dados_tratados.csv')

    # Mapeamento de códigos numéricos para rótulos textuais
    niveis_educacao_map = {
        -1: 'Desconhecido', 0: 'Sem Escolaridade', 1: 'Primário',
        2: 'Ensino Médio', 3: 'Superior'
    }
    df['education_label'] = df['education'].map(niveis_educacao_map)

    marital_status_map = {0: 'Solteira', 1: 'Casada', -1: 'Desconhecido'}
    df['marital_status_label'] = df['marital_status'].map(marital_status_map)
    
    # ADIÇÃO: Mapeamento para Situação Profissional
    employment_map = {0: 'Desempregada', 1: 'Parcialmente Empregada', 2: 'Empregada', -1: 'Desconhecido'}
    df['employment_label'] = df['employment'].map(employment_map)
    
    df['violence_label'] = df['violence'].map({True: 'Sim', False: 'Não'})

    # Categorização da variável 'income' em faixas
    df['income_category'] = pd.cut(
        df['income'],
        bins=[-1, 1000, 2000, 3000, float('inf')],
        labels=["Até 1k", "1k-2k", "2k-3k", "3k+"]
    )
    return df

# Executa o carregamento dos dados
df_original = carregar_dados()

# --- 3. BARRA LATERAL COM FILTROS ---

st.sidebar.header("Opções de Filtragem")

# Filtro por Nível de Escolaridade
educacao_selecionada = st.sidebar.multiselect(
    "Nível de Escolaridade:",
    options=df_original['education_label'].unique(),
    default=df_original['education_label'].unique()
)

# ADIÇÃO: Filtro por Situação Profissional
situacao_selecionada = st.sidebar.multiselect(
    "Situação Profissional:",
    options=df_original['employment_label'].unique(),
    default=df_original['employment_label'].unique()
)

# ADIÇÃO: Filtro por Estado Civil
estado_civil_selecionado = st.sidebar.multiselect(
    "Estado Civil:",
    options=df_original['marital_status_label'].unique(),
    default=df_original['marital_status_label'].unique()
)

# Filtro por Faixa de Idade
idade_min, idade_max = int(df_original['age'].min()), int(df_original['age'].max())
faixa_idade = st.sidebar.slider(
    "Faixa de Idade:",
    min_value=idade_min,
    max_value=idade_max,
    value=(idade_min, idade_max)
)

# Aplicação dos filtros ao DataFrame
df_filtrado = df_original[
    (df_original['education_label'].isin(educacao_selecionada)) &
    (df_original['age'].between(faixa_idade[0], faixa_idade[1])) &
    (df_original['employment_label'].isin(situacao_selecionada)) & 
    (df_original['marital_status_label'].isin(estado_civil_selecionado))
]

# --- 4. CONTEÚDO PRINCIPAL DO DASHBOARD ---

st.title("Dashboard para Análise de Dados sobre Violência Doméstica")

st.markdown("""
* **Disciplina:** Ciências de Dados 35T34
* **Docente:** Luiz Affonso Henderson Guedes de Oliveira
* **Discentes:** Ramylla Luiza Barbalho Gomes Bezerra, Juan Pablo Hortencio Ferreira

Este painel interativo apresenta uma análise exploratória de um conjunto de dados sobre violência doméstica.
Utilize os filtros na barra lateral para segmentar os dados. A interatividade dos gráficos permite a visualização
de valores detalhados ao passar o cursor sobre os elementos.
""")

# --- Métricas Principais ---
st.header("Métricas Principais")

if not df_filtrado.empty:
    total_casos = df_filtrado.shape[0]
    casos_violencia = df_filtrado['violence'].sum()
    taxa_violencia = (casos_violencia / total_casos) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Respondentes", f"{total_casos}")
    col2.metric("Casos de Violência Reportados", f"{casos_violencia}")
    col3.metric("Taxa de Violência", f"{taxa_violencia:.2f}%")
else:
    st.warning("Nenhum dado corresponde aos filtros selecionados.")


# --- Análise Gráfica ---
st.header("Análise Gráfica")

# Paleta de cores original (Vermelho para Sim, Azul para Não)
PALETA_ORIGINAL = {'Sim': '#d62728', 'Não': '#1f77b4'}

# Layout de gráficos em duas colunas
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("Incidência de Violência por Estado Civil")
    fig1 = px.histogram(
        df_filtrado,
        x='marital_status_label',
        color='violence_label',
        barmode='group',
        title='Contagem de Casos por Estado Civil',
        labels={'marital_status_label': 'Estado Civil', 'violence_label': 'Ocorrência de Violência', 'count': 'Contagem'},
        color_discrete_map=PALETA_ORIGINAL
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_graf2:
    st.subheader("Incidência de Violência por Faixa de Renda")
    fig2 = px.histogram(
        df_filtrado,
        x='income_category',
        color='violence_label',
        barmode='group',
        title='Contagem de Casos por Faixa de Renda',
        labels={'income_category': 'Faixa de Renda', 'violence_label': 'Ocorrência de Violência', 'count': 'Contagem'},
        category_orders={"income_category": ["Até 1k", "1k-2k", "2k-3k", "3k+"]},
        color_discrete_map=PALETA_ORIGINAL
    )
    st.plotly_chart(fig2, use_container_width=True)


# ADIÇÃO: Novo layout de colunas para os próximos gráficos
col_graf3, col_graf4 = st.columns(2)

with col_graf3:
    st.subheader("Distribuição de Idades na Amostra")
    fig3 = px.histogram(
        df_filtrado,
        x='age',
        nbins=20,
        title='Histograma da Distribuição de Idades',
        labels={'age': 'Idade', 'count': 'Contagem'},
        color_discrete_sequence=['#1f77b4'] 
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_graf4:
    # ADIÇÃO: Novo gráfico de Situação Profissional
    st.subheader("Incidência de Violência por Situação Profissional")
    fig4 = px.histogram(
        df_filtrado,
        x='employment_label',
        color='violence_label',
        barmode='group',
        title='Contagem de Casos por Situação Profissional',
        labels={'employment_label': 'Situação Profissional', 'violence_label': 'Ocorrência de Violência', 'count': 'Contagem'},
        color_discrete_map=PALETA_ORIGINAL
    )
    st.plotly_chart(fig4, use_container_width=True)


st.subheader("Matriz de Correlação entre Variáveis")
st.markdown("A análise de correlação a seguir foi calculada sobre o conjunto de dados completo para fornecer uma visão geral das relações lineares entre as variáveis numéricas.")

correlation_matrix = df_original.drop(columns=['id']).corr(numeric_only=True)
fig5 = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu',
    title='Mapa de Calor da Correlação de Pearson'
)
st.plotly_chart(fig5, use_container_width=True)

# Exibição opcional da tabela de dados
if st.checkbox("Exibir tabela com dados filtrados"):
    st.dataframe(df_filtrado)

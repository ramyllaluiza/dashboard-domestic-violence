#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Violência Doméstica",
    layout="wide"
)

# --- 2. CARREGAMENTO E PROCESSAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dados_tratados.csv')
    niveis_educacao_map = {-1: 'Desconhecido', 0: 'Sem Escolaridade', 1: 'Primário', 2: 'Ensino Médio', 3: 'Superior'}
    df['education_label'] = df['education'].map(niveis_educacao_map)
    marital_status_map = {0: 'Solteira', 1: 'Casada', -1: 'Desconhecido'}
    df['marital_status_label'] = df['marital_status'].map(marital_status_map)
    employment_map = {0: 'Desempregada', 1: 'Parc. Empregada', 2: 'Empregada', -1: 'Desconhecido'}
    df['employment_label'] = df['employment'].map(employment_map)
    df['violence_label'] = df['violence'].map({True: 'Sim', False: 'Não'})
    df['income_category'] = pd.cut(df['income'], bins=[-1, 1000, 2000, 3000, float('inf')], labels=["Até 1k", "1k-2k", "2k-3k", "3k+"])
    return df

# Executa o carregamento dos dados
df_original = carregar_dados()

# --- 3. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("Dashboard")
    
    with st.expander("Informações do Projeto"):
        st.markdown("""
        **Disciplina:** Ciências de Dados 35T34
        * **Docente:** Luiz Affonso Henderson Guedes de Oliveira
        * **Discentes:** Ramylla Luiza Barbalho Gomes Bezerra, Juan Pablo Hortencio Ferreira
        """)

    with st.expander("Filtros Sociodemográficos", expanded=False):
        estado_civil_selecionado = st.multiselect("Estado Civil:", options=df_original['marital_status_label'].unique(), default=df_original['marital_status_label'].unique())
        idade_min, idade_max = int(df_original['age'].min()), int(df_original['age'].max())
        faixa_idade = st.slider("Faixa de Idade:", min_value=idade_min, max_value=idade_max, value=(idade_min, idade_max))

    with st.expander("Filtros Socioeconômicos", expanded=False):
        educacao_selecionada = st.multiselect("Nível de Escolaridade:", options=df_original['education_label'].unique(), default=df_original['education_label'].unique())
        situacao_selecionada = st.multiselect("Situação Profissional:", options=df_original['employment_label'].unique(), default=df_original['employment_label'].unique())
    
# Aplicação dos filtros ao DataFrame
df_filtrado = df_original[
    (df_original['education_label'].isin(educacao_selecionada)) &
    (df_original['age'].between(faixa_idade[0], faixa_idade[1])) &
    (df_original['employment_label'].isin(situacao_selecionada)) & 
    (df_original['marital_status_label'].isin(estado_civil_selecionado))
]

# --- 4. CONTEÚDO PRINCIPAL DO DASHBOARD ---
st.header("Análise Interativa de Violência Doméstica")
st.caption(f"Violência doméstica contra mulheres em uma área rural específica de um país em desenvolvimento.")
st.caption(f"Última atualização: 17 de junho de 2025")
st.divider()

# --- CARD DE MÉTRICAS PRINCIPAIS ---
with st.container(border=True):
    st.subheader("Resumo dos dados")
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

# --- SEÇÃO DE ANÁLISE GRÁFICA ---
st.header("Análise Gráfica Detalhada")

PALETA_AZUL_VERMELHO = {'Sim': '#DC143C', 'Não': '#4682B4'} 

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
    # MUDANÇA: Usando um azul mais forte e adicionando contorno
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

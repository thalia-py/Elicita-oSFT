# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 09:34:42 2025

@author: Thalia
"""

import streamlit as st
import numpy as np
import random as rd
from scipy.stats import weibull_min
from scipy.integrate import dblquad

# Funções de densidade
def fz(z, forma, escala):
    return (forma / escala) * ((z / escala) ** (forma - 1)) * np.exp(-((z / escala) ** forma))

def fh(h, d_medio):
    return (1 / d_medio) * np.exp(- (1 / d_medio) * h)

# Interface Streamlit
st.title("📊 Elicitação para o Modelo Delay-Time")

st.markdown("Preencha os parâmetros abaixo para estimar os modelos de falha e de defeito.")

unidade = st.text_input("Unidade de tempo (ex.: horas, dias, semanas):", key="unidade")
TM = st.number_input(f"Tempo médio até a falha do sistema ({unidade})", min_value=0.0, format="%.2f", key="TM")
DM = st.number_input(f"Tempo médio de atraso (janela de detecção do defeito) ({unidade}):", min_value=0.0, format="%.2f", key="DM")
ID = st.number_input("Imprecisão na estimativa do tempo de atraso (%)", min_value=0.0, format="%.2f", key="ID")


# Opções qualitativas
opcoes_qualitativas = {
    "Muito Alta (quase certa) – 90 a 100%": (90, 99.99),
    "Alta – 75 a 90%": (75, 90),
    "Média-Alta – 60 a 75%": (60, 75),
    "Média – 40 a 60%": (40, 60),
    "Média-Baixa – 25 a 40%": (25, 40),
    "Baixa – 10 a 25%": (10, 25),
    "Muito Baixa (improvável) – 0 a 10%": (0.01, 10),
}

# Coleta das probabilidades
if TM > 0:
    pontos_tempo = [int(x) for x in [0.1 * TM, 0.5 * TM, TM, 1.5 * TM, 2 * TM]]
    matriz_t = np.zeros(5)
    matriz_p = np.zeros([5, 2])

    st.subheader("📌 Probabilidades de Sobrevivência Estimadas")
    for i in range(5):
        t = pontos_tempo[i]
        resposta = st.selectbox(f"Chance do sistema continuar operacional após {t} {unidade}:", list(opcoes_qualitativas.keys()), key=f"resp_{i}")
        matriz_t[i] = t
        matriz_p[i][0] = opcoes_qualitativas[resposta][0]
        matriz_p[i][1] = opcoes_qualitativas[resposta][1]

st.divider()

# Cálculo com botão
if st.button("▶️ Estimar Parâmetros"):
    # Monte Carlo para Weibull de tempo até falha (Z)
    amostra_eta = np.zeros(1000)
    amostra_beta = np.zeros(1000)

    for i in range(1000):
        f_prob = np.zeros(5)
        s_prob = np.zeros(5)
        s_prob[0] = rd.uniform(matriz_p[0][0], matriz_p[0][1])
        f_prob[0] = (100 - s_prob[0]) / 100
        for j in range(1, 5):
            lim_sup = min(s_prob[j - 1], matriz_p[j][1])
            s_prob[j] = rd.uniform(matriz_p[j][0], lim_sup)
            f_prob[j] = (100 - s_prob[j]) / 100

        x_input = np.log(matriz_t)
        y_input = np.log(-np.log(1 - f_prob))
        A, B = np.polyfit(x_input, y_input, 1)
        beta = A
        eta = np.exp(-B / beta)
        amostra_beta[i] = beta
        amostra_eta[i] = eta

    beta_25 = np.percentile(amostra_beta, 25)
    beta_75 = np.percentile(amostra_beta, 75)
    beta_central = (beta_25 + beta_75) / 2
    beta_imprecisao = 100 * (beta_75 - beta_central) / beta_central
    eta_25 = np.percentile(amostra_eta, 25)
    eta_75 = np.percentile(amostra_eta, 75)
    eta_central = (eta_25 + eta_75) / 2
    eta_imprecisao = 100 * (eta_75 - eta_central) / eta_central

    st.subheader("📉 Parâmetros Weibull – Tempo até a Falha")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Forma (β)", f"{beta_central:.2f}", f"IQR: {beta_25:.2f} – {beta_75:.2f}")
        st.metric("Imprecisão relativa β", f"{beta_imprecisao:.2f}%")
    with col2:
        st.metric("Escala (η)", f"{eta_central:.2f} {unidade}", f"IQR: {eta_25:.2f} – {eta_75:.2f}")
        st.metric("Imprecisão relativa η", f"{eta_imprecisao:.2f}%")

    # Estimativa para X = Z - H via convolução invertida
    amostra_eta_x = np.zeros(100)
    amostra_beta_x = np.zeros(100)

    for k in range(100):
        escala = rd.uniform(eta_25, eta_75)
        forma = rd.uniform(beta_25, beta_75)
        d_medio = rd.uniform((1 - ID / 100) * DM, (1 + ID / 100) * DM)

        x_input = np.zeros(30)
        y_input = np.zeros(30)
        ini = 0
        for i in range(30):
            ini += 0.1
            t = ini * escala
            x_input[i] = np.log(t)
            prob = 1 - dblquad(lambda h, z: fz(z, forma, escala) * fh(h, d_medio),
                               t, 10 * escala,
                               lambda z: 0, lambda z: z - t)[0]
            y_input[i] = np.log(np.log(1 / (1 - prob)))

        A, B = np.polyfit(x_input, y_input, 1)
        beta_x = A
        eta_x = np.exp(-B / beta_x)
        amostra_eta_x[k] = eta_x
        amostra_beta_x[k] = beta_x

    beta_x_25 = np.percentile(amostra_beta_x, 25)
    beta_x_75 = np.percentile(amostra_beta_x, 75)
    beta_x_central = (beta_x_25 + beta_x_75) / 2
    beta_x_imprecisao = 100 * (beta_x_75 - beta_x_central) / beta_x_central
    eta_x_25 = np.percentile(amostra_eta_x, 25)
    eta_x_75 = np.percentile(amostra_eta_x, 75)
    eta_x_central = (eta_x_25 + eta_x_75) / 2
    eta_x_imprecisao = 100 * (eta_x_75 - eta_x_central) / eta_x_central

    st.subheader("🔧 Parâmetros Weibull – Tempo até o Defeito")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Forma (β)", f"{beta_x_central:.2f}", f"IQR: {beta_x_25:.2f} – {beta_x_75:.2f}")
        st.metric("Imprecisão relativa β", f"{beta_x_imprecisao:.2f}%")
    with col4:
        st.metric("Escala (η)", f"{eta_x_central:.2f} {unidade}", f"IQR: {eta_x_25:.2f} – {eta_x_75:.2f}")
        st.metric("Imprecisão relativa η", f"{eta_x_imprecisao:.2f}%")

# =============================================================================
# Rodapé
# =============================================================================
st.markdown(""" 
<hr style="border:0.5px solid #333;" />

<div style='color: #aaa; font-size: 13px; text-align: left;'>
    <strong style="color: #ccc;">RANDOM - Grupo de Pesquisa em Risco e Análise de Decisão em Operações e Manutenção</strong><br>
    Criado em 2012, o grupo reúne pesquisadores dedicados às áreas de risco, manutenção e modelagem de operações.<br>
    <a href='http://random.org.br' target='_blank' style='color:#888;'>Acesse o site do RANDOM</a>
</div>

""", unsafe_allow_html=True)

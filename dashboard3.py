# -*- coding: utf-8 -*-
"""
DASHBOARD INTEGRADO — JV Peças (v8)
-----------------------------------
Correções para cálculo exato do total de vendas com base na coluna "Preço venda" (valor unitário)
e formatação brasileira completa.
Inclui KPIs: Total de Vendas, Itens Vendidos, Ticket Médio e Receita dos Modelos.
"""

import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard Integrado — JV Peças", layout="wide")

# ==========================
# Leitura robusta
# ==========================

def read_csv_smart(uploaded_file):
    raw = uploaded_file.read()
    encodings = ["utf-8", "latin-1", "cp1252"]
    seps = [",", ";"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(io.StringIO(raw.decode(enc, errors="strict")), sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    raise RuntimeError("Não foi possível ler o CSV.")

# ==========================
# Funções auxiliares
# ==========================

PREFIXOS = ["BIZ","CG","NXR","XRE","CRF","CBR","CB","PCX","TRX","POP","FOURTRAX"]
RX_GERAL = re.compile(r"\b([A-Z]{2,7})\s*[-/]?\s*(\d{2,3})\b")
RX_COLADO_LETRA = re.compile(r"\b([A-Z]{2,7})(\d{2,3})[A-Z]\b")
RX_COLADO_SEM_LETRA = re.compile(r"\b([A-Z]{2,7})(\d{2,3})\b")

def extract_model(text):
    if not isinstance(text, str):
        return None
    t = text.upper()
    if "FOURTRAX" in t:
        return "TRX 420"
    if "BROS" in t and "160" in t:
        return "NXR 160"
    m = RX_GERAL.search(t)
    if m and m.group(1) in PREFIXOS:
        return f"{m.group(1)} {m.group(2)}"
    m2 = RX_COLADO_LETRA.search(t)
    if m2 and m2.group(1) in PREFIXOS:
        return f"{m2.group(1)} {m2.group(2)}"
    m3 = RX_COLADO_SEM_LETRA.search(t)
    if m3 and m3.group(1) in PREFIXOS:
        return f"{m3.group(1)} {m3.group(2)}"
    for pref in PREFIXOS:
        p = t.find(pref)
        if p != -1:
            around = t[p:p+16]
            m4 = re.search(r"(\d{2,3})", around)
            if m4:
                return f"{pref} {m4.group(1)}"
    return None

def to_float_br(x):
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip()
    s = re.sub(r"[^0-9,.-]", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def br_currency(v: float) -> str:
    s = f"{float(v):,.2f}"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")

def pick_column(df, candidates, default=None):
    cols_upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        for U, c in cols_upper.items():
            if cand in U:
                return c
    return default

# ==========================
# Interface
# ==========================

st.title("📦 Dashboard Integrado de Vendas de Peças — E-commerce")
st.markdown("Envie o arquivo **original** (.csv). Agora com cálculo exato e formatação brasileira.")

uploaded = st.file_uploader("Envie o arquivo CSV de vendas", type=["csv"])
if uploaded is None:
    st.info("⬆️ Envie o arquivo para começar.")
    st.stop()

with st.spinner("Processando dados..."):
    df = read_csv_smart(uploaded)
    df.columns = [c.strip() for c in df.columns]

    # Detecta colunas relevantes
    col_prod = pick_column(df, ["NOME PROD", "PRODUTO", "DESCRI", "ITEM", "NOME"])
    col_qtd = pick_column(df, ["QTD", "QUANT", "QUANTIDADE"])
    col_preco = pick_column(df, ["PREÇO VENDA", "PRECO VENDA", "VALOR", "PREÇO", "PRECO"])

    if not col_preco:
        st.error("Não encontrei coluna de preço de venda. Verifique os nomes.")
        st.write("Colunas:", list(df.columns))
        st.stop()

    df["__produto_upper"] = df[col_prod].astype(str).str.upper()
    df["MODELO"] = df["__produto_upper"].apply(extract_model)
    df["__QTD"] = pd.to_numeric(df[col_qtd], errors="coerce").fillna(1.0) if col_qtd else 1.0
    df["__VALOR"] = df[col_preco].apply(to_float_br).fillna(0.0)
    df["__TOTAL_LINHA"] = df["__VALOR"] * df["__QTD"]

    base = df.dropna(subset=["MODELO"]).copy()
    receita_modelo = base.groupby("MODELO")["__TOTAL_LINHA"].sum().sort_values(ascending=False).reset_index(name="__RECEITA")
    qtd_modelo = base.groupby("MODELO")["__QTD"].sum().sort_values(ascending=False).reset_index()

    def remove_modelo_do_nome(nome, modelo):
        if isinstance(nome, str) and isinstance(modelo, str):
            nome = nome.replace(modelo, "")
        nome = re.sub(r"\(REF\.[^)]+\)", "", str(nome))
        nome = re.sub(r"\s+", " ", str(nome)).strip()
        return nome

    base["__PECA_SEM_MODELO"] = [remove_modelo_do_nome(n, m) for n, m in zip(base["__produto_upper"], base["MODELO"])]
    pecas_freq = (
        base.groupby(["MODELO", "__PECA_SEM_MODELO"]).size().reset_index(name="OCORRENCIAS")
        .sort_values(["MODELO", "OCORRENCIAS"], ascending=[True, False])
    )

# ==========================
# KPIs principais
# ==========================

valor_total_vendas = float(df["__TOTAL_LINHA"].sum())
itens_totais_vendidos = float(df["__QTD"].sum())
ticket_medio = valor_total_vendas / itens_totais_vendidos if itens_totais_vendidos else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Valor Total de Vendas (R$)", br_currency(valor_total_vendas))
col2.metric("📦 Itens Vendidos", f"{int(itens_totais_vendidos):,}")
col3.metric("🎟️ Ticket Médio (R$)", br_currency(ticket_medio))
col4.metric("🏍️ Modelos Identificados", len(receita_modelo))

st.metric("🧾 Receita (apenas modelos)", br_currency(receita_modelo["__RECEITA"].sum()))

# ==========================
# Gráficos
# ==========================

st.header("🏍️ Modelos com Maior Receita e Quantidade Vendida")
colA, colB = st.columns(2)

if not receita_modelo.empty:
    fig_receita = px.bar(receita_modelo.head(10), x='MODELO', y='__RECEITA', title='Top 10 Modelos por Receita', text_auto='.2s', color='MODELO')
    colA.plotly_chart(fig_receita, use_container_width=True)

if not qtd_modelo.empty:
    fig_qtd = px.bar(qtd_modelo.head(10), x='MODELO', y='__QTD', title='Top 10 Modelos por Quantidade Vendida', text_auto='.2s', color='MODELO')
    colB.plotly_chart(fig_qtd, use_container_width=True)

st.header('🔩 Peças mais frequentes por modelo')
modelos = sorted(pecas_freq['MODELO'].dropna().unique())
modelo_sel = st.selectbox('Escolha o modelo para detalhar as peças', modelos) if len(modelos) else None

if modelo_sel:
    pecas_modelo = pecas_freq[pecas_freq['MODELO'] == modelo_sel].head(15)
    if not pecas_modelo.empty:
        fig_pecas = px.bar(pecas_modelo, x='__PECA_SEM_MODELO', y='OCORRENCIAS', title=f'Peças mais frequentes — {modelo_sel}', text_auto=True, color='OCORRENCIAS')
        fig_pecas.update_layout(xaxis_title='Peça', yaxis_title='Ocorrências', showlegend=False)
        st.plotly_chart(fig_pecas, use_container_width=True)

st.header('📋 Tabela completa de peças por modelo')
st.dataframe(pecas_freq, use_container_width=True)


import streamlit as st
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ouvidoria Inteligente", layout="wide")

@st.cache_resource
def carregar_modelo(nome_modelo):
    return SentenceTransformer(nome_modelo)

@st.cache_data
def carregar_dados():
    with open('manifestacoes.json', 'r', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

@st.cache_data
def gerar_embeddings(textos, _modelo):
    return _modelo.encode(textos)

# Configurações Sidebar
st.sidebar.title("Configurações do Sistema")
escolha_modelo = st.sidebar.selectbox("Modelo", ["paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-small-pt-v1.5"])
top_k = st.sidebar.slider("Top-K Manifestações", 1, 10, 5)

df = carregar_dados()
modelo = carregar_modelo(escolha_modelo)
embeddings = gerar_embeddings(df['texto'].tolist(), modelo)

aba1, aba2, aba3, aba4 = st.tabs(["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"])

with aba1:
    st.subheader("Buscador Semântico de Manifestações")
    query = st.text_input("Descreva o problema reclamado:")
    if query:
        query_emb = modelo.encode([query])
        sims = cosine_similarity(query_emb, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        for idx in top_indices:
            score = sims[idx]
            cor = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
            st.markdown(f"**{cor} Score: {score:.2f} | Categoria: {df.iloc[idx]['categoria_oficial']}**")
            st.write(df.iloc[idx]['texto'])

with aba2:
    st.subheader("Base Completa de Manifestações")
    st.dataframe(df)
    if st.button("Gerar Matriz de Similaridade"):
        matriz = cosine_similarity(embeddings)
        fig, ax = plt.subplots()
        sns.heatmap(matriz, cmap="mako", xticklabels=False, yticklabels=False)
        st.pyplot(fig)

with aba3:
    st.subheader("Visualização do Espaço Vetorial 2D (PCA)")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    df_pca = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'Categoria': df['categoria_oficial']})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_pca, x='x', y='y', hue='Categoria', ax=ax, s=100)
    st.pyplot(fig)
    st.info("Comentário do Aluno: É possível visualizar a formação de agrupamentos temáticos que, em sua maioria, coincidem com os rótulos oficiais, ainda que exista uma sobreposição natural entre 'Meio Ambiente' e 'Infraestrutura'.")

with aba4:
    st.subheader("Teste de Estratégias de Chunking")
    texto_longo = st.text_area("Insira um texto longo (>500 palavras):")
    col1, col2 = st.columns(2)
    tamanho = col1.number_input("Chunk Size", value=250)
    sobreposicao = col2.number_input("Chunk Overlap", value=50)
    
    if st.button("Dividir Texto") and texto_longo:
        splitter = RecursiveCharacterTextSplitter(chunk_size=tamanho, chunk_overlap=sobreposicao)
        chunks = splitter.split_text(texto_longo)
        for i, c in enumerate(chunks):
            st.success(f"**Chunk {i+1}**:\n{c}")
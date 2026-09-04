import streamlit as st
import numpy as np
import google.generativeai as genai
import os

# Configuração da página do Streamlit
st.set_page_config(page_title="Simulador e Agente - Apostas", page_icon="📊")
st.title("Agente Educacional: Análise de Risco em Apostas")

# Configuração do Gemini (Requer que a GEMINI_API_KEY esteja nos Secrets do Streamlit)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Chave de API não encontrada. Configure os Secrets do Streamlit.")
    st.stop()

# 1. Definição do Motor Matemático
def simular_monte_carlo(odd: float, aposta: float, repeticoes: int, banca_inicial: float) -> dict:
    """
    Simula trajetórias de retornos financeiros esperados para apostas de quota fixa.
    Obrigatório acionar esta ferramenta quando o usuário perguntar sobre projeções, lucros ou riscos financeiros.
    """
    if odd <= 1.0:
        return {"erro": "A odd (cotação) deve ser estritamente maior que 1.0"}
    
    p = 1.0 / odd
    lucro_liquido = aposta * (odd - 1.0)
    ev = (p * lucro_liquido) - ((1.0 - p) * aposta)
    
    n_sims = 10000
    vitorias = np.random.binomial(n=repeticoes, p=p, size=n_sims)
    derrotas = repeticoes - vitorias
    
    bancas_finais = banca_inicial + (vitorias * lucro_liquido) - (derrotas * aposta)
    prob_ruina = np.sum(bancas_finais <= 0) / n_sims
    
    return {
        "valor_esperado_por_aposta": round(ev, 2),
        "probabilidade_de_ruina_percentual": round(float(prob_ruina) * 100, 2),
        "banca_media_esperada": round(float(np.mean(bancas_finais)), 2)
    }

# 2. Configuração do System Prompt e Base de Conhecimento
instrucao_sistema = """
Você é um assistente educacional sobre o mercado de apostas esportivas (bets) no Brasil.
Seu objetivo é explicar conceitos de economia comportamental (aversão à perda, falácia do jogador, desconto hiperbólico) e estatística.
Regras estritas:
1. Nunca recomende apostas ou forneça estratégias para ganhar.
2. Para qualquer pergunta envolvendo projeções, retornos ou risco financeiro ao longo do tempo, você DEVE usar a ferramenta simular_monte_carlo.
3. Explique os resultados da simulação de forma clara, conectando o valor esperado negativo com a vantagem da casa (house edge).
"""

# Inicializa o modelo apenas uma vez
@st.cache_resource
def carregar_modelo():
    return genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[simular_monte_carlo],
        system_instruction=instrucao_sistema
    )

modelo = carregar_modelo()

import os

# [...] (Mantenha os imports, a função simular_monte_carlo e a instrucao_sistema iguais)

@st.cache_resource
def inicializar_agente_com_documentos():
    """
    Executa o upload dos PDFs apenas uma vez na inicialização do servidor
    e injeta os objetos de arquivo no histórico inicial do chat.
    """
    arquivos_gemini = []
    diretorio_conhecimento = "conhecimento"
    
    # 1. Faz o upload da base bibliográfica (Kahneman, relatórios SPA, etc.)
    if os.path.exists(diretorio_conhecimento):
        for arquivo in os.listdir(diretorio_conhecimento):
            if arquivo.endswith(".pdf"):
                caminho = os.path.join(diretorio_conhecimento, arquivo)
                doc = genai.upload_file(path=caminho)
                arquivos_gemini.append(doc)
    
    # 2. Declara a LLM com a ferramenta de simulação acoplada
    modelo = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[simular_monte_carlo],
        system_instruction=instrucao_sistema
    )
    
    # 3. Inicializa o histórico. A primeira mensagem oculta carrega os tensores dos PDFs.
    historico_inicial = []
    if arquivos_gemini:
        historico_inicial.append({
            "role": "user",
            "parts": arquivos_gemini
        })
        # Força uma mensagem de confirmação do modelo para estabilizar o histórico
        historico_inicial.append({
            "role": "model",
            "parts": ["Documentos carregados. Usarei esta base para análises comportamentais."]
        })

    chat = modelo.start_chat(
        history=historico_inicial,
        enable_automatic_function_calling=True
    )
    return chat

# 3. Gerenciamento de Estado da Sessão (Histórico do Chat)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = inicializar_agente_com_documentos()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
    st.session_state.mensagens.append({
        "role": "assistant", 
        "content": "Olá. Sou o assistente de avaliação de risco estatístico. Como posso ajudar na sua análise hoje?"
    })

# [...] (Mantenha a Renderização da Interface e o Processamento do Input idênticos)


# 5. Processamento do Input do Usuário
if prompt := st.chat_input("Digite sua dúvida ou os parâmetros da aposta..."):
    # Exibe a mensagem do usuário
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Aciona o modelo e exibe a resposta
    with st.chat_message("assistant"):
        with st.spinner("Processando simulação estocástica e análise estrutural..."):
            try:
                resposta = st.session_state.chat_session.send_message(prompt)
                st.markdown(resposta.text)
                st.session_state.mensagens.append({"role": "assistant", "content": resposta.text})
            except Exception as e:
                st.error(f"Erro na execução da matriz de processamento: {e}")

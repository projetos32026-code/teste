import streamlit as st
import numpy as np
from openai import OpenAI
import json
import os

st.set_page_config(page_title="Simulador e Agente - Apostas", page_icon="📊")
st.title("Agente Educacional: Análise de Risco em Apostas")

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("Chave de API não encontrada. Configure os Secrets do Streamlit em .streamlit/secrets.toml ou na Cloud.")
    st.stop()

# 1. Base de Conhecimento Estática (Substituindo os PDFs para simplificar a arquitetura)
def carregar_base_teorica():
    # Em produção, você pode usar open('texto.txt').read() aqui
    return """
    Referencial Teórico:
    - Aversão à perda: A dor de perder é psicologicamente mais intensa que o prazer de ganhar.
    - Falácia do jogador: A crença irracional de que eventos passados afetam eventos independentes futuros.
    - Expectativa matemática nas apostas esportivas sempre favorece a plataforma devido à margem da casa (house edge).
    """

# 2. Definição do Motor Matemático
def simular_monte_carlo(odd: float, aposta: float, repeticoes: int, banca_inicial: float) -> str:
    """Função Python pura. Retorna JSON stringificado para o LLM."""
    if odd <= 1.0:
        return json.dumps({"erro": "A odd (cotação) deve ser estritamente maior que 1.0"})
    
    p = 1.0 / odd
    lucro_liquido = aposta * (odd - 1.0)
    ev = (p * lucro_liquido) - ((1.0 - p) * aposta)
    
    n_sims = 10000
    vitorias = np.random.binomial(n=repeticoes, p=p, size=n_sims)
    derrotas = repeticoes - vitorias
    
    bancas_finais = banca_inicial + (vitorias * lucro_liquido) - (derrotas * aposta)
    prob_ruina = np.sum(bancas_finais <= 0) / n_sims
    
    return json.dumps({
        "valor_esperado_por_aposta": round(ev, 2),
        "probabilidade_de_ruina_percentual": round(float(prob_ruina) * 100, 2),
        "banca_media_esperada": round(float(np.mean(bancas_finais)), 2)
    })

# 3. Mapeamento da Ferramenta (Schema OpenAI)
tools = [
    {
        "type": "function",
        "function": {
            "name": "simular_monte_carlo",
            "description": "Simula trajetórias de retornos financeiros e calcula o risco de falência para apostas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "odd": {"type": "number", "description": "A odd decimal da aposta (ex: 1.5)"},
                    "aposta": {"type": "number", "description": "Valor financeiro apostado por rodada"},
                    "repeticoes": {"type": "integer", "description": "Número de vezes que a aposta será repetida"},
                    "banca_inicial": {"type": "number", "description": "Saldo financeiro inicial do usuário"}
                },
                "required": ["odd", "aposta", "repeticoes", "banca_inicial"]
            }
        }
    }
]

# 4. Gerenciamento de Estado da Sessão
instrucao_sistema = f"""
Você é um assistente educacional sobre o mercado de apostas esportivas no Brasil.
Baseie-se neste conteúdo para responder: {carregar_base_teorica()}
Regras estritas:
1. Nunca recomende apostas.
2. Para perguntas envolvendo projeções, risco ou simulações financeiras ao longo do tempo, VOCÊ DEVE obrigatoriamente usar a ferramenta simular_monte_carlo.
3. Explique os resultados da simulação citando heurísticas comportamentais.
"""

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {"role": "system", "content": instrucao_sistema},
        {"role": "assistant", "content": "Olá. Sou o assistente de avaliação de risco estatístico. Como posso ajudar na sua análise?"}
    ]

# Renderização da Interface (Ignorando mensagens de sistema e chamadas de ferramenta brutas)
for msg in st.session_state.mensagens:
    if msg["role"] not in ["system", "tool"] and not msg.get("tool_calls"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 5. Loop de Processamento e Roteamento
if prompt := st.chat_input("Digite sua dúvida ou os parâmetros da aposta..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Processando simulação e analisando contexto..."):
            # Primeira chamada ao LLM
            resposta_inicial = client.chat.completions.create(
                model="gpt-4o-mini", # Modelo com excelente custo-benefício para agentes
                messages=st.session_state.mensagens,
                tools=tools,
                tool_choice="auto"
            )
            
            mensagem_retorno = resposta_inicial.choices[0].message
            
            # Condicional de roteamento: O LLM decidiu usar o simulador?
            if mensagem_retorno.tool_calls:
                # O histórico exige que a intenção de chamada da ferramenta seja anexada
                st.session_state.mensagens.append(mensagem_retorno)
                
                for tool_call in mensagem_retorno.tool_calls:
                    if tool_call.function.name == "simular_monte_carlo":
                        # Extrai os parâmetros inferidos pelo LLM
                        args = json.loads(tool_call.function.arguments)
                        resultado_str = simular_monte_carlo(
                            odd=args.get("odd"),
                            aposta=args.get("aposta"),
                            repeticoes=args.get("repeticoes"),
                            banca_inicial=args.get("banca_inicial")
                        )
                        
                        # Anexa o resultado determinístico no histórico como papel "tool"
                        st.session_state.mensagens.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "simular_monte_carlo",
                            "content": resultado_str
                        })
                
                # Segunda chamada ao LLM para sintetizar os números do simulador
                resposta_sintese = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.mensagens
                )
                
                texto_final = resposta_sintese.choices[0].message.content
                st.markdown(texto_final)
                st.session_state.mensagens.append({"role": "assistant", "content": texto_final})
            
            else:
                # O LLM decidiu responder direto (ex: dúvida estritamente teórica)
                st.markdown(mensagem_retorno.content)
                st.session_state.mensagens.append({"role": "assistant", "content": mensagem_retorno.content})

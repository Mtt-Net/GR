# Configuração do estilo dos gráficos
plt.style.use('default')
sns.set_theme()

# CSS customizado para Streamlit
st.markdown(
    """
    <style>
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    .info-box p {
        margin: 10px 0;
        line-height: 1.5;
        text-align: justify;
    }
    .info-box h3 {
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .info-box strong {
        color: #2980b9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Definições em HTML estilizado
st.markdown(
    """
    <h2>🎯 Calculadora de Opções</h2>
    <div class="info-box" style="margin-bottom: 20px;">
        <h3>📚 Definições</h3>
        <p><strong>Opções Europeias:</strong> São contratos financeiros que conferem ao detentor o direito, mas não a obrigação,
        de comprar (opção de compra) ou vender (opção de venda) um ativo por um preço previamente determinado apenas na data
        específica de vencimento da opção.</p>
        <p><strong>Opções Asiáticas:</strong> São instrumentos financeiros derivados nos quais o pagamento depende da média
        dos preços do ativo subjacente durante um determinado período, reduzindo, assim, o impacto de oscilações extremas
        no preço em um único dia.</p>
        <p><strong>Opção Call:</strong> É um contrato que garante ao seu titular o direito, sem a obrigação, de comprar
        um ativo (como ações) por um preço fixo até uma data específica, sendo vantajosa se o preço do ativo subir acima
        desse valor combinado.</p>
        <p><strong>Opção Put:</strong> É um contrato que dá ao seu titular o direito, também sem obrigação, de vender
        um ativo por um preço fixado até uma data determinada, sendo vantajosa se o preço do ativo cair abaixo desse
        valor combinado.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar para parâmetros
st.sidebar.header('Parâmetros da Opção')
ticker = st.sidebar.text_input('Ticker', value='AAPL')
tipo_opcao = st.sidebar.radio('Tipo de Opção', ['Opção Europeia', 'Opção Asiática'])
option_style = st.sidebar.radio('Direito', ['Call', 'Put'])
strike_multiplier = st.sidebar.slider('Strike (% do S0)', 0.8, 1.2, 1.05, 0.01)
tempo_expiracao = st.sidebar.slider('Tempo até Expiração (anos)', 0.1, 2.0, 1.0, 0.1)
taxa_livre_risco = st.sidebar.slider('Taxa Livre de Risco', 0.01, 0.15, 0.04, 0.01)

@st.cache_data(show_spinner=False)
def capturar_parametros(ticker, periodo='1y'):
    dados = yf.download(ticker, period=periodo)
    dados['Retornos'] = np.log(dados['Close'] / dados['Close'].shift(1))
    dados = dados.dropna()
    S0 = dados['Close'].iloc[-1].item()
    mu = dados['Retornos'].mean() * 252
    sigma = dados['Retornos'].std() * np.sqrt(252)
    return S0, mu, sigma, dados

def monte_carlo_opcao_europeia_call(S0, K, T, r, sigma, n_sim=10000):
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_opcao_europeia_put(S0, K, T, r, sigma, n_sim=10000):
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_opcao_asiatica_call(S0, K, T, r, sigma, n_sim=10000, n_steps=252):
    dt = T / n_steps
    payoffs = []
    for _ in range(n_sim):
        prices = [S0]
        for _ in range(n_steps):
            Z = np.random.normal()
            St = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            prices.append(St)
        media = np.mean(prices)
        payoffs.append(max(media - K, 0))
    return np.exp(-r * T) * np.mean(payoffs)

def monte_carlo_opcao_asiatica_put(S0, K, T, r, sigma, n_sim=10000, n_steps=252):
    dt = T / n_steps
    payoffs = []
    for _ in range(n_sim):
        prices = [S0]
        for _ in range(n_steps):
            Z = np.random.normal()
            St = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            prices.append(St)
        media = np.mean(prices)
        payoffs.append(max(K - media, 0))
    return np.exp(-r * T) * np.mean(payoffs)

enviar = st.button('Calcular')

if enviar:
    try:
        S0, mu, sigma, dados = capturar_parametros(ticker)
        K = S0 * strike_multiplier
        T = tempo_expiracao
        r = taxa_livre_risco

        if tipo_opcao == 'Opção Europeia':
            if option_style == 'Call':
                price = monte_carlo_opcao_europeia_call(S0, K, T, r, sigma)
            else:
                price = monte_carlo_opcao_europeia_put(S0, K, T, r, sigma)
        else:
            if option_style == 'Call':
                price = monte_carlo_opcao_asiatica_call(S0, K, T, r, sigma)
            else:
                price = monte_carlo_opcao_asiatica_put(S0, K, T, r, sigma)

        st.markdown(f'''
        <div class="result-container">
            <h3>📊 Resultados para {ticker}</h3>
            <div class="info-box">
                <p><strong>Tipo:</strong> {tipo_opcao} ({option_style})</p>
                <p><strong>Preço Atual (S0):</strong> R$ {S0:.2f}</p>
                <p><strong>Strike (K):</strong> R$ {K:.2f}</p>
                <p><strong>Volatilidade (σ):</strong> {sigma:.2%}</p>
                <p><strong>Preço da Opção:</strong> R$ {price:.2f}</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        dados['Close'].plot(ax=ax1)
        ax1.set_title(f"Histórico de Preço - {ticker}")
        ax1.set_xlabel("Data")
        ax1.set_ylabel("Preço")
        ax1.grid(True, alpha=0.3)
        dt = T / 252
        for _ in range(50):
            prices = [S0]
            for _ in range(252):
                Z = np.random.standard_normal()
                prices.append(prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z))
            ax2.plot(prices, alpha=0.4)
        ax2.set_title("Simulação Monte Carlo - Trajetórias de Preço")
        ax2.set_xlabel("Dias")
        ax2.set_ylabel("Preço")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Erro ao processar o ticker: {str(e)}")

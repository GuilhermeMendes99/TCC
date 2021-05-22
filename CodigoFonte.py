##
## ESTRATÉGIAS DE INVESTIMENTO NO MERCADO DE AÇÕES BRASILEIRO BASEADAS EM VIESES COMPORTAMENTAIS ##
##

# Importa as bibliotecas necessárias

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import statsmodels.api as sm


# Classe para baixar dados
class MarketData():
    
    # Inicialiar
    def __init__(self):
        
        self.index_path = r"C:\" # Caminho do arquivo com os dados dos Índices
        self.close_path = r"C:\" # Caminho do arquivo com os dados do preço de fech. das ações
        self.volume_path = r"C:\" # Caminho do arquivo com os dados do volume das ações
        self.skiprows = 143 # Quantas linhas pular para começar a ler as bases de dados (define período de início da análise)
        
        # Inicializar bases de dados
        self.init_data()
        
    # Inicializar dados
    def init_data(self):
        
        # Carrega as bases de dados do Índice, do fechamento e volume das ações, e do CDI
        self.index = self.load_data(self.index_path, names=['Data', 'IBrX'], usecols=[0,3])
        self.close = self.load_data(self.close_path, stockdata=True)
        self.volume = self.load_data(self.volume_path, stockdata=True)
        self.cdi = self.load_data(self.index_path, names=['Data', 'CDI'], usecols=[0,1])
        
        
    # Função para carregar as bases de dados
    # path : caminho do diretório
    # stockdata : (T|F) verdadeiro para dados de ações, falso para índice
    def load_data(self, path, stockdata=False, **kwargs):

        # Carregar o arquivo .xlsx escolhido
        data = pd.read_excel(path, index_col=0, header=3, **kwargs)[self.skiprows:]

        # Corrige formatação de datas
        data.index = [dt.datetime.strptime(date, "%b-%y") for date in data.index]

        # Se forem dados de ações
        if stockdata==True:
            # Corrige nomes das colunas
            data.columns = [column[-5:] for column in data.columns]

            # Substitui valores em branco por 0
            data = data.replace(to_replace="-", value=0)

        # Retorna os dados obtidos
        return data


# Classe para rodar o modelo
class MarketModel():
    
    # Inicializar
    def __init__(self, fp, hp, data, inverse=False):
        
        self.fp = fp # Formation period
        self.hp = hp # Holding period
        
        self.data = data # dados
        self.n_stocks = 10 # Número de ações selecionadas a cada mês
        self.setup_data() # Organizar dados
        
        self.inverse = inverse # Se verdadeiro, a ponta long do long short será a da estratégia contrária
        
        # Rodar
        self.run()
        
    # Organizar dados
    def setup_data(self):
        
        # Carrega as bases de dados do Índice, do fechamento e volume das ações, e do CDI
        self.index = self.data.index
        self.close = self.data.close
        self.volume = self.data.volume
        self.cdi = self.data.cdi
        
    # Normalizar portfólios
    def normalize(self, portfolio):
        
        d = len(portfolio) - len(self.w)
        
        return portfolio[d:] / portfolio.iloc[d].values[0]
    
    
    # Função que retorna, a partir dos DataFrames 'close' e 'volume', apenas as ações que estão no top decil de volume negociado,
    # para um dado período i (índice do DataFrame)
    def top_decile(self,i):

        # Calcula o volume necessário para estar no primeiro decil de ações mais negociadas neste período
        decile_cut = np.percentile(self.volume.iloc[i], q=90)

        # Verifica quais ações negociaram pelo menos este valor de corte no período e retorna
        return self.close.loc[:, self.volume.ge(decile_cut).iloc[i].values]
    
    # Função que constrói o histórico de posições do portfólio
    # type : 0 = Winner, 1 = Loser
    def build_portfolio(self,ptype=0):

        # Inicia com um array vazio
        stocks = []

        # Cria outro array para salvar as ações sem overlap de períodos
        stocks_n_overlap = []

        # Repetir do período fp até o último período
        for i in range(self.fp, len(self.volume)):

            # Define a amostra a partir do decil de ações negociadas com maior volume
            sample = self.top_decile(i)

            # Considera apenas as ações que foram negociadas em todos os meses no período i a i-fp
            sample = sample.loc[:, (sample.iloc[[i,i-self.fp]] != 0).all(axis=0)]

            # Considera o retorno destas ações no período
            sample = sample.iloc[i] / sample.iloc[i-self.fp].values - 1

            # Ordena os retornos em ordem crescente
            sample = pd.DataFrame(columns=['Retorno'], index=sample.index, data=sample.values)
            sample = sample.sort_values(by="Retorno")

            # W deste período é o conjunto das 'n' últimas ações (maior retorno passado), e L as 'n' primeiras (menor retorno)
            # Define entre retornar W ou L a partir do argumento ptype, associando à variável 'new_stocks'
            if ptype==0:
                new_stocks = sample.index[-self.n_stocks:].values
            elif ptype==1:
                new_stocks = sample.index[:self.n_stocks].values

            # Adiciona as ações deste período ao array sem overlap
            stocks_n_overlap.append(new_stocks)

            # Se já houver no mínimo hp portfólios formados
            if len(stocks_n_overlap) >= self.hp:

                # Constrói o array para adicionar as ações deste período com overlap dos períodos anteriores, com base no
                # período de permanência (hp)
                add_stocks = []

                for t in range(0,self.hp):

                    # Adiciona as ações do período i-t
                    add_stocks = [*add_stocks, *stocks_n_overlap[i-self.fp-t]]

                # Adiciona todas as ações consideradas a 'stocks'
                stocks.append(add_stocks)

        # Converte em DataFrame para manipulação
        stocks = pd.DataFrame(data=stocks, index=self.close.index[(self.fp+self.hp-1):])

        return stocks

    
    # Custo de transação
    def get_tc(self):
        return -0.01 / self.hp
    
    
    # Função que retorna o histórico (evolução de $1 investido) de um portfólio, em que este é definido a partir de
    # um DataFrame com o conjunto de ações daquele portfólio em cada período do tempo
    def portfolio_history(self,portfolio):

        # Inicializa as variáveis
        money = 1
        money_history = [money]

        # Para cada período,
        for k in range(0, len(portfolio)-1):

            # Inicializa as variáveis
            factor = 0
            valid_stocks = 0

            # Para cada ação no portfólio neste período
            for ticker in portfolio.iloc[k].values:
                
                # Se o ticker for válido
                if type(ticker) == str:
                    
                    period = k+self.fp+self.hp
                
                    ap = self.close[[ticker]].iloc[period].values[0] # Preço de hoje
                    lp = self.close[[ticker]].iloc[period-1].values[0] # Preço anterior

                    # Se houver dados de preço de fechamento desta ação no período atual e no período anterior
                    if ap != 0 and lp != 0:

                        # Adiciona ao fator de rentabilidade o retorno desta ação no período
                        factor = factor + (ap / lp - 1)

                        # Aumenta a contagem do número de ações
                        valid_stocks = valid_stocks + 1

            # O retorno no período é dado pela média do retorno das ações válidas do portfólio
            factor = factor / valid_stocks + 1
            
            # Retira da rentabilidade os custos de transação            
            factor = factor * (1 + self.get_tc())

            # Atualiza o histórico de $
            money = money * factor
            money_history.append(money)

        # Converte o histórico em DataFrame e retorna
        money_history = pd.DataFrame(data=money_history, index=self.close.index[(self.fp+self.hp-1):])

        return money_history
    
    # Retorna o histórico do portfólio long-short
    def longshort_history(self):
        
        # Inicializa as variáveis
        money = 1
        money_history = [money]
        
        # Para cada período
        for k in range(1,len(self.w)):
            
            # Calcula a diferença de retornos (long-short) atualizada pelo CDI
            fl = (self.w.iloc[k].values[0] / self.w.iloc[k-1].values[0])
            fs = (self.l.iloc[k].values[0] / self.l.iloc[k-1].values[0])
            fc = (self.n_cdi.iloc[k].values[0] / self.n_cdi.iloc[k-1].values[0])
            
            if self.inverse == True:                
                factor = fs / fl * fc
            else:
                factor = fl / fs * fc
            
            # Retira custos de transação
            short_cost = 1.01 ** (1/12) - 1
            factor = factor * (1 + 2 * (self.get_tc() - short_cost))
                               
            # Atualiza o histórico de $
            money = money * factor
            money_history.append(money)
                               
        # Converte o histórico em DataFrame e retorna
        money_history = pd.DataFrame(data=money_history, index=self.w.index)

        return money_history

    
    # Função que roda a regressão dos retornos do portfólio contra os do benchmark
    def regression(self, portfolio):

        # Calcula a taxa de retorno livre de risco
        rf = self.n_cdi.pct_change()[1:].values

        # Calcula o excesso de retorno do portfolio em relação ao ativo livre de risco
        p_returns = portfolio.pct_change()[1:].subtract(rf)
        p_returns.columns = ['Portfolio']

        # Calcula o prêmio de risco
        erp = self.n_index.pct_change()[1:].subtract(rf)
        erp = sm.add_constant(erp)
        erp.columns = ['Alpha', 'Beta']

        # Modela p_returns em função do erp
        model = sm.OLS(p_returns, erp)
        results = model.fit()

        return results
    
    # Função que calcula o retorno total anualizado do portfólio
    def ann_return(self, port):
        
        ann_return = ((port.iloc[len(port)-1].values[0]) ** (12/len(port)) - 1) * 100
        
        return ann_return
    
    
    # Função para plotagem de resultados
    def plot(self, portfolios, names, dtype=['-', '--']):
        
        # Inicializa o gráfico
        fig, ax = plt.subplots(figsize=(11,6))
        plt.yscale('log')

        # Para cada portfolio
        for p in range(0,len(portfolios)):
            
            # Plota o gráfico
            port = portfolios[p]        
            ax.plot(port, dtype[p], label=names[p], color='black', linewidth=1)
                
        # Detalhes do gráfico
        ax.grid(color='lightgrey')
        ax.legend()

        # Mostra o gráfico
        plt.show()
        
        
    # Função para executar o programa
    def run(self):

        # Constrói os portfólios W e L
        self.winners = self.build_portfolio(0)
        self.losers = self.build_portfolio(1)

        # Calcula a evolução do $ investido em cada portfólio no período
        self.w = self.portfolio_history(self.winners)
        self.l = self.portfolio_history(self.losers)
        
        # Índices normalizados
        self.n_cdi = self.normalize(self.cdi)
        self.n_index = self.normalize(self.index)
        
        # Long Short
        self.ls = self.longshort_history()

# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

#lendo arquio csv
#o pandas gera um dataframe na leitura do csv. Por isso 'df'
df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

#gerando dummies a partir das variavies categóricas
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

#porcentagem da massa de dados que será utilizada para treino
porcentagem_treino = 0.9

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

#pegando os primeiros valores do array equivalentes a 90% definidos acima.
#no nosso caso 900 valores para treino
treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

#pegando os ultimos valores do array equivalentes a 10% da massa de dados.
#no nosso caso 100 valores para teste
teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]



#utilizando metodo fit para treinar o algoritmo utilizando o 
#algoritmo multinomial de naive_bayes... 'MultinomialNB'
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

#método 'predict' tenta prever o resultado baseado no que o algotimo aprendeu
resultado = modelo.predict(teste_dados)

#comparando o resultado obtido pelo algoritmo com o resultado real de massa de dados
diferencas = resultado - teste_marcacoes

#para cada resultado correto do algoritmo, insere no array de acertos
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

#verificando porcentagem de acertos
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print 'Porcentagem de acerto ' + str(taxa_de_acerto) + '%'
print 'Elementos testados ' + str(total_de_elementos)
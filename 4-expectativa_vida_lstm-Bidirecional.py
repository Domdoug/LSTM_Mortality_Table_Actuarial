#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar bibliotecas
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import os
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
#from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

import time
import warnings
warnings.filterwarnings("ignore")


# ### Modelagem com a utilização do pacote embeding para Bi-direcional
# ##### 
# Os humanos não começam a pensar do zero a cada segundo. Ao ler este ensaio, você entende cada palavra com base na sua compreensão das palavras anteriores. Você não joga tudo fora e começa a pensar do zero novamente. Seus pensamentos têm persistência.
# 
# As redes neurais tradicionais não podem fazer isso, e isso parece uma grande falha. Por exemplo, imagine que você queira classificar que tipo de evento está acontecendo em todos os momentos do filme. Não está claro como uma rede neural tradicional poderia usar seu raciocínio sobre eventos anteriores do filme para informar os posteriores.
# 
# Redes neurais recorrentes abordam esse problema. São redes com loops, permitindo que as informações persistam.
# 
# Em comparação com o LSTM, BLSTMou BiLSTMpossui duas redes, uma pastinformação de acesso na forwarddireção e outro acesso futurena reversedireção.
# 
# LSTMse suas variantes bidirecionais são populares porque tentaram aprender como e quando esquecer e quando não usar portas em sua arquitetura. Em RNNarquiteturas anteriores , o desaparecimento de gradientes era um grande problema e fazia com que essas redes não aprendessem muito.
# 
# Usando Bidirecional LSTMs, você alimenta o algoritmo de aprendizado com os dados originais uma vez do começo ao fim e uma vez do fim ao começo. Existem debates aqui, mas geralmente ele aprende mais rápido que a abordagem unidirecional, embora dependa da tarefa.
# 

# #### 1 - carregar base tratada

# In[2]:


# Verifica a pasta corrente
pasta = os.getcwd()


# In[3]:


pasta_resultados = os.path.join(pasta, "resultados")
pasta_graficos = os.path.join(pasta, "graficos1")


# In[ ]:


# Regex notation by "\s+". This means a single space, or multiple spaces are all to be treated as a single separator.
# df_dados = pd.read_csv('bltper_1x1.txt', skiprows=2, sep = '\s+') 
df_dados = pd.read_csv(os.path.join(pasta, "dados") + "/" + 'bltper_1x1.txt', skiprows=2, sep = '\s+') 


# In[ ]:


df_dados.head().append(df_dados.tail())


# #### 2 - criar features, entre elas a logqx, que corresponde já convertido a escala logarítma da probabilidade de morte
# 

# In[ ]:


# Tratamento da idade 110+ para os anos.
# DataFrame.loc[condition, column_name] = new_value
df_dados.loc[(df_dados.Age == '110+'),'Age'] = 110


# In[ ]:


# Criar a feature log qx
df_dados['logqx'] = np.log(df_dados['qx'])
# Aproveitar e corrigir a tipagem da feature Age
df_dados["Age"] = df_dados["Age"].astype(int)


# In[ ]:


df_dados.head().append(df_dados.tail())


# In[ ]:


df_dados.shape


# #### 3 - Criar a feature tempo t, com base na feature ano, que corresponde ao elemento temporal da série

# In[8]:


# Preparar dataset
#serie = {'t': ano, 'logqx_prob': logqx_prob}
df_lstm = pd.DataFrame(df_dados, columns=['Age','Year','logqx'])
df_lstm['t'] = pd.to_datetime(df_lstm['Year'], format='%Y')
#df_lstm.drop(['ano'], axis=1, inplace=True)
df_lstm.set_index('t', inplace=True)


# In[9]:


#df_lstm[df_lstm['t'].dt.year == 1998]
df_lstm.head()


# In[10]:


df_lstm[df_lstm['Age']==0]


# #### 4 - Separar a base em base de treino e base de teste para cada idade x ao longo dos anos t, ou seja, para a idade 0, entre os anos 1998 a 2018, para a idade 1, no mesmo período e assim por diante.

# ##### Rotina LSTM, métricas e gráficos

# In[12]:


# proximos testes: n_epochs = 1000 e base teste de 3 anos. Sugestão. No programa em script. 
# Usar, também, o código para salvar o log do compilie
predict_res = []
pred_actual_rmse_res = []

w_max = max(df_dados['Age']) # definir maior idade nas tábuas. testes: 3

# inicio do cronometro do processamento
start = time.time()

n_input = 30 # 10 # Length of the output sequences (in number of timesteps). Corresponde ao número de dados 
# que usaremos para a rede. No caso, 10 anos na idade = 0, 10 anos na idade=1, etc.Vamos testar com 3 anos??
n_features = 1 # Número de features, variáveis. O modelo é univariavel (qx) para cada idade.
n_epochs = 500 # 1000 #500
n_batch = 2  # Number of timeseries samples in each batch (except maybe the last one).
n_neurons = 50
t_projecao = 30
# (#batch_size,#inputs,#features) 3D
for x in range(0, w_max+1):
    
    # Série para cada idade ao longo dos anos de 1998 a 2018
    #serie = df_lstm[df_lstm['idade']==x]['logqx_prob']
    serie = df_lstm[df_lstm['Age']==x]
    serie.drop(['Age', 'Year'], axis=1, inplace=True)

    # Separar base de treino e teste === preparar dados
    treino, teste = serie[:-30], serie[-30:]
    
    # Padronizar dados: Normalizar entre 0 e 1
    scaler = MinMaxScaler()
    scaler.fit(treino)
    treino = scaler.transform(treino)
    teste = scaler.transform(teste)
    
    #generator = TimeseriesGenerator(treino, treino, length=n_input, batch_size=n_batch)
    # length: The number of lag observations to use in the input portion of each sample (e.g. 3)
    # That is the desired number of lag observations to use as input = VAmos tentar 21: 2018-1998
    
    # batch_size: The number of samples to return on each iteration (e.g. 32)
    # The samples are not shuffled by default. This is useful for some recurrent neural networks
    # like LSTMs that maintain state across samples within a batch.
    # both the data and target for this generator is “treino”.
    
    generator = TimeseriesGenerator(treino, treino, length=n_input, batch_size=n_batch)
    
    # ============================ CAMADAS =========== CAMADAS =================================
    # A camada LSTM já possui, em sua construção, funções default de ativação:
    # activation="tanh",recurrent_activation="sigmoid",
    # três funções sigmoide e 1 tangente hiperbólica
    
    # Modelo bidirecional
    model = Sequential()
    model.add(Bidirectional(LSTM(n_neurons, return_sequences=True), input_shape=(n_input, n_features)))
    model.add(Bidirectional(LSTM(n_neurons)))
    model.add(Dropout(0.20))
    model.add(Dense(1))
    #model.add(Activation('softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(optimizer='adam', loss='mse', metrics=["mae"])
    
    
    
    
    #model = Sequential()
    
    # #reshape the data into LSTM required (#batch,#timesteps,#features)
    #Adding the first LSTM layer and some Dropout regularisation
    
    #model.add(LSTM(n_neurons, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    #model.add(LSTM(n_neurons, activation='relu', input_shape=(n_input, n_features)))
    #model.add(Dropout(0.20))
    
    # Adding a second LSTM layer and some Dropout regularisation
    #model.add(LSTM(n_neurons))
    #model.add(Dropout(0.20))
    
    # Adding a third LSTM layer and some Dropout regularisation
    #model.add(LSTM(n_neurons, return_sequences=True))
    #model.add(Dropout(0.20))

    # Adding a fourth LSTM layer and some Dropout regularisation
    #model.add(LSTM(n_neurons))
    #model.add(Dropout(0.20))
    
    # Adding the output layer
    #model.add(Dense(1))
    
    # ============================ CAMADAS =========== CAMADAS =================================
    
    #model.add(Dense(y.shape[1], activation='sigmoid'))
    #model.compile(optimizer='adam', loss='mse', metrics=["mae"])
    
    
    # fit model
    #model.fit_generator(generator, epochs=n_epochs)
    
    # ADAPTADO PARA A ATUALIZAÇÃO DO KERAS (28/11/2020)
    csv_logger = CSVLogger('log_modelo_demography_bidirecional.csv', append=True, separator=';')
    model.fit(generator, epochs=n_epochs, callbacks=[csv_logger])
    # model.fit(X_train, Y_train, callbacks=[csv_logger])
    
    #Previsão
    pred_list = []
    batch = treino[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_input):  # n_input
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:,1:,:], [[pred_list[i]]], axis=1)

    #inverse transform forecasts and test. Need to scale them back so we can compare the final results
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list), 
                            index=serie[-n_input:].index, columns=['Prediction'])
    
    df_teste = pd.concat([serie, df_predict], axis=1)

    # Gráfico da estimativa, com a base de teste
    #plt.figure(figsize=(10,5))
    '''
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111) 
    ax.plot(df_teste.index, df_teste['logqx'])
    ax.plot(df_teste.index, df_teste['Prediction'], color='r')
    ax.legend(loc='best', fontsize='xx-large', labels=['logqx', 'Estimativa'])
    fig.suptitle('logqx Bi-Direcional LSTM dataset teste na idade = %i' %x, fontweight="bold") 
    plt.savefig(pasta_graficos + '/' + 'prev_Bi_Direcional_LSTM_test_idade'+str(x)+'.png')
    '''
    
    pred_actual_rmse = rmse(df_teste.iloc[-n_input:, [0]], df_teste.iloc[-n_input:, [1]])
    print("idade:", x, "rmse: ", pred_actual_rmse)
    
    
    pred_actual_rmse_res.append(pred_actual_rmse)
    
    treino = serie
    
    scaler.fit(treino)
    treino = scaler.transform(treino)
    
    #generator = TimeseriesGenerator(treino, treino, length=n_input, batch_size=n_batch)
    #model.fit_generator(generator,epochs=n_epochs)
    # ADAPTADO PARA A ATUALIZAÇÃO DO KERAS (28/11/2020)
    # length: The number of lag observations to use in the input portion of each sample (e.g. 3)
    # batch_size: The number of samples to return on each iteration (e.g. 32)
    
    generator = TimeseriesGenerator(treino, treino, length=n_input, batch_size=n_batch)
    
    #model.fit(generator, epochs=n_epochs, batch_size=n_batch)
    model.fit(generator, epochs=n_epochs)
        
    pred_list = []

    batch = treino[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_input):   
        pred_list.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
          
    # prever para t_projecao anos
    add_dates = [serie.index[-1] + DateOffset(years=x) for x in range(0, t_projecao + 1)]
    #add_dates = [serie.index[-1] + pd.offsets.YearBegin(x) for x in range(0,6)]
    future_dates = pd.DataFrame(index=add_dates[1:],columns=serie.columns)
    
    
    #inverse transform forecasts and test. Need to scale them back so we can compare the final results
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

    
    predict_res.append(df_predict.values.tolist())
    
    df_proj = pd.concat([serie,df_predict], axis=1)
    
    '''
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)  # "111" means "1x1 grid, first subplot"
    ax.plot(df_proj.index, df_proj['logqx'])
    ax.plot(df_proj.index, df_proj['Prediction'], color='r')
    ax.legend(loc='best', fontsize='xx-large', labels=['logqx', 'Predição'])
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    fig.suptitle('Logqx  Bi-Direcional LSTM projetado na idade = %i' %x, fontweight = "bold") # Título parametrizado com a idade
    
    plt.savefig(pasta_graficos + '/' + 'proj_Bi_Direcional_LSTM_log_qx'+str(x)+'.png')
    
    '''

# fim do cronometro do processamento

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print()
print('Tempo de processamento:')
print('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
print()

#Tempo de processamento: 22:32:14.93

# Tempo de processamento sem imprimir os gráficos: 18:35:29.37


# #### 5 - Valores de RMSE por idade

# In[13]:


pd.DataFrame(pred_actual_rmse_res) # RMSE para cada idade


# In[52]:


#### 5 - Base resultante dos anos de 2019 a 2028, por idade


# In[14]:


df_lstm_res = pd.DataFrame(predict_res)


# In[15]:


df_lstm_res.head()


# In[16]:


df_lstm_res[0][0][0]


# In[17]:


df_lstm_res.info()


# In[18]:


# Função para unir as listas em linha
def unirSeries(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


# In[19]:


colunas = np.arange(2020, 2050)
df_temp = pd.DataFrame(predict_res, columns=colunas)
df_lstm_res = unirSeries(df_temp,colunas)
df_lstm_res = df_lstm_res.reset_index(drop=True)


# In[20]:


df_lstm_res.head()


# In[21]:


df_forecast_res_exp = pd.DataFrame(np.exp(df_lstm_res))


# In[22]:


df_forecast_res_exp.head()


# In[ ]:


# Gravar resultados


# In[23]:


df_forecast_res_exp.to_csv(pasta_resultados + '/' + 'lstm_previsao_qx_500_Bi_Direcional_demography.csv')


# In[24]:


pd.DataFrame(pred_actual_rmse_res).to_csv(pasta_resultados + '/' + 'pred_actual_rmse_res_500_Bi_Direcional_demography.csv', header=['RMSE'])


# In[ ]:





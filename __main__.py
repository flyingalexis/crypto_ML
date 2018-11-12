import pandas as pd
import os
from dataProcess import cnn_preprocess,sequential_preprocess
from models import cnn_model,lstm_model
from models import params_optimizer as p_o
import decimal

# load csv
# from utils import database
# df = database.getDataDf('XRPUSD','1m')
# df.to_csv('{0}_{1}.csv'.format('XRPUSD','1m'))
# read csv

df = pd.DataFrame.from_csv('datasets/crypto_hist/{0}_{1}.csv'.format('XRPUSD','1m'))

# ---------------------------------------LSTM----------------------------------
data_x, data_y, idx ,loop_cond = sequential_preprocess.data_preprocess(df)
print(data_x.shape)
# m_lstm = lstm_model.lstm(data_x,data_y)
# m_lstm.run_network(data_x,data_y)
# while loop_cond:
#     data_x, data_y, idx ,loop_cond = sequential_preprocess.data_preprocess(df,idx = idx)
#     m_lstm.run_network(data_x,data_y)

# ============================== data feeding  ===============================================
# data_X , data_Y= cnn_preprocess.data_preprocess(df,'D')
# print('finish data processing')
# cnn = cnn_model.cnn(data_X,data_Y,{'lr': decimal.Decimal('0.1'), 'filter_num': (144, 136, 72, 128), 'layers_num': 4})
# cnn.optimize()    # method use for train the model

# ========================================== EOS ===============================================

# ========================================= KCV ================================================

# p_o.cnn_optimize(data_X, data_Y)

# ========================================== EOS ===============================================
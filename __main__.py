import pandas as pd
import os
from dataProcess import csv_process,cnn_preprocess
from models import cnn_model
from models import params_optimizer as p_o
import decimal

df = csv_process.merge_csv('hsi')
data_X , data_Y= cnn_preprocess.data_preprocess(df,'D')
print('finish data processing')


# ============================== data feeding  ===============================================

cnn = cnn_model.cnn(data_X,data_Y,{'lr': decimal.Decimal('0.1'), 'filter_num': (144, 136, 72, 128), 'layers_num': 4})
cnn.optimize()    # method use for train the model

# ========================================== EOS ===============================================

# ========================================= KCV ================================================

# p_o.cnn_optimize(data_X, data_Y)

# ========================================== EOS ===============================================
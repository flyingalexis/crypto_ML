import pandas as pd
import os
from dataProcess import csv_process,cnn_preprocess
from models import cnn_model

df = csv_process.merge_csv('hsi')
data_X , data_Y= cnn_preprocess.data_preprocess(df,'D')
print('finish data processing')
print('Feeding data into model !!')
cnn = cnn_model.cnn(data_X,data_Y)
cnn.optimize()

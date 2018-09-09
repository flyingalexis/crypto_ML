import pandas as pd
import os
from dataProcess import csv_process,cnn_preprocess

df = csv_process.merge_csv('hsi')
cnn_preprocess.data_preprocess(df,'D')


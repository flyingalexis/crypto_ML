from pymongo import MongoClient
import pandas as pd

CLIENT = MongoClient('localhost', 27017)
DB = CLIENT['stock_db']
COLLECTIONS = {
  'K_day' : DB['K_day'],
  'K_m15' : DB['K_M15'] # and a list of lines that we need to analyze
  }

def record_k_line(df, k_type):
  dic = df.to_dict("records")
  COLLECTIONS[k_type].insert_many(dic)

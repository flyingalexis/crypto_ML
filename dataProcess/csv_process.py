import os
import pandas as pd
'''
this method is to merge csv 
output to /datasets/processed/[stock_code]
returns a df
'''
def merge_csv(stock_code):
  path = os.getcwd() + '/datasets/' + stock_code + '/'
  arr_of_dfs = []
  for filename in os.listdir(path):
      arr_of_dfs.append(pd.read_csv(path + filename , delimiter='\t'))
  df = pd.concat(arr_of_dfs).drop_duplicates().reset_index(drop=True)
  df['Date'] = pd.to_datetime(df.Date)
  df.sort_values('Date', ascending=True,inplace = True) # This now sorts in date order
  df.to_csv(stock_code + '.csv')
  return df
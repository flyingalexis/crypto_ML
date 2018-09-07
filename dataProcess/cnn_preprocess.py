import pandas as pd
'''
  return the 
'''
def data_preprocess(df,freq):
  df_list = []  # list to store daily stock price dfs
  df['Close'].diff().std()
  i = 1
  a = []
  a.append(df.groupby(pd.Grouper(key='Date',freq=freq)).apply(lambda sub_df : sub_df if sub_df.size != 0 else None))
  print(len(a))
  print('alfie')
  print(len(df_list))
  df_list[0].to_csv('demo0.csv')
  print(df_list[0]['Close'])

  for sub_df in df_list:  
    # print(sub_df.size)
      # print(sub_df)
    print(sub_df['Close'].size)
    sub_df['Close'] = sub_df['Close'] - sub_df['Close'].iloc[0]

  # df_list[0].to_csv('demo1.csv')
  # return df_list
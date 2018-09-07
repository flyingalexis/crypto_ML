import pandas as pd
'''
  return the (list of features, list of labels)
'''
def data_preprocess(df,freq):
  df_list = []  # list to store daily stock price dfs
  df_y_list = [] # label list 
  sd_factor = df['Close'].diff().std()  # take sd of close in one year for the denorminator of standardization
  for daily_df in df.groupby(pd.Grouper(key='Date',freq=freq)):
    if daily_df[1].size != 0:
      df_list.append(daily_df[1])
  close_min = 0 
  close_max = 0
  init_prices = []
  for sub_df in df_list:
    init_prices.append(sub_df['Close'].iloc[0]/sd_factor)
    sub_df[['Open','High','Low','Close']] =  (sub_df[['Open','High','Low','Close']] - sub_df['Close'].iloc[0]) / sd_factor
    close_max = sub_df['Close'].max() if sub_df['Close'].max() > close_max else close_max
    close_min = sub_df['Close'].min() if sub_df['Close'].min() < close_min else close_min
  scale_range = close_max - close_min
  for idx ,sub_df in enumerate(df_list):
    if idx > 0 :
      df_y_list.append((sub_df[['Open','High','Low','Close']] + init_prices[idx] - init_prices[idx-1] - close_min) / scale_range)
      # print(df_y_list[idx-1][['Open','High','Low','Close']].mean(axis=0).tolist())
    sub_df[['Open','High','Low','Close']] = (sub_df[['Open','High','Low','Close']] - close_min) / scale_range 
  return df_list,df_y_list

#   Open     0.613377
# High     0.624280
# Low      0.601345
# Close    0.614603
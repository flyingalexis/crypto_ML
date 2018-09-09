import pandas as pd
'''
  return the (list of features, list of labels)
  which a label is % of timesteps of the label day will be higher than the last close value of input date and the 
'''
def data_preprocess(df,freq):
  df_list = []  # list to store daily stock price dfs
  df_x_list = []
  df_y_list = [] # label list 
  norm_factor = 0
  for daily_df in df.groupby(pd.Grouper(key='Date',freq=freq)):
    if daily_df[1].size != 0:
      df_list.append(daily_df[1]) # split datasets by date
  for idx,sub_df in enumerate(df_list):    # get max change range within every 2 day range (make sure all stock price data lies within 0-1)
    if idx < len(df_list) -1 :
      max = sub_df['High'].max() if sub_df['High'].max() > df_list[idx + 1]['High'].max() else df_list[idx + 1]['High'].max()
      min = sub_df['Low'].min() if sub_df['Low'].min() < df_list[idx + 1]['Low'].max() else df_list[idx + 1]['Low'].min()
    norm_factor = max - min if max - min > norm_factor else norm_factor
  for idx ,sub_df in enumerate(df_list):
    if idx < len(df_list) -1 :
      # y_df = ( (df_list[idx + 1][['Open','High','Low','Close']] - sub_df['Close'].iloc[0]) / norm_factor / 2 ) + 0.5
      y_df = ( (df_list[idx + 1][['Close']] - sub_df['Close'].iloc[0]) / norm_factor / 2 ) + 0.5
      x_df = ( (sub_df[['Open','High','Low','Close']] - sub_df['Close'].iloc[0]) / norm_factor / 2 ) + 0.5
      df_x_list.append(x_df)
      labels = (y_df[y_df >= x_df.iloc[0]].count() / y_df.shape[-1]).tolist()
      labels.append(y_df['Close'].mean())
      df_y_list.append(labels)
  return df_x_list,df_y_list

# Open     0.613377
# High     0.624280
# Low      0.601345
# Close    0.614603
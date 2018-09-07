  path = './datasets/hsi/'

  arr_of_dfs = []
  for filename in os.listdir(path):
      arr_of_dfs.append(pd.read_csv(path + filename , delimiter='\t'))

  df = pd.concat(arr_of_dfs).drop_duplicates().reset_index(drop=True)
  df['Date'] =pd.to_datetime(df.Date)
  df.sort_values('Date', ascending=False,inplace = True) # This now sorts in date order
  df.to_csv('merged.csv')

  print(df)
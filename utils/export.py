def export_csv_pandas(df, name):
  df.to_csv(name +".csv", sep=',', encoding='utf-8')
  
def data_preprocess(df, x_seq_len = 20, y_seq_len = 20):
    max = df['High'].max()
    min = df['Low'].min()
    norm_fac = max - min 
    print('max {} , {} , {}', max ,min ,norm_fac)
    p_df = (df [['High','Low','Open','Close']] - min) / norm_fac 
    print(p_df)
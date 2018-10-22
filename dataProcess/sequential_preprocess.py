import pandas as pd
import numpy as np

def data_preprocess(df, x_seq_len = 1440, y_seq_len = 120, stride = 20):
    max = df['high'].max()
    min = df['low'].min()
    norm_fac = max - min 
    vol_min = df['volume'].min()
    vol_norm_fac = df['volume'].max() - vol_min
    p_df = (df [['high','low','open','close']] - min) / norm_fac 
    v_df = (df['volume'] - vol_min ) / vol_norm_fac
    result = pd.concat([p_df, v_df], axis=1, sort=False)
    if result.shape[0] < x_seq_len + y_seq_len:
        raise Exception('Data not enough')
    
    x = None
    y = None

    for i in range(x_seq_len + y_seq_len  ,result.shape[0],stride):
        print(i)
        x_start = i - x_seq_len - y_seq_len
        x_end = y_start= i - y_seq_len
        y_end = i
        x_df = result.iloc[x_start:x_end,]
        y_df = result.iloc[y_start:y_end,]
        y_min = y_df[['high','low','open','close']].min().values
        y_max = y_df[['high','low','open','close']].max().values
        y_mean = y_df[['high','low','open','close']].mean().values
        y_data = np.concatenate( (y_min,y_max,y_mean), axis = 0 )
        if x is None:
            x = x_df.values[None,:,:]
        else:
            x = np.concatenate([x, x_df.values[None,:,:]], axis=0)
        
        if y is None:
            y = y_data[None,:]
        else:
            y = np.concatenate([y, y_data[None,:]], axis=0)

        if i >= 50000:
            break

    return x,y

    # print(result)

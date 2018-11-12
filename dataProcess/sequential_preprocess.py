import pandas as pd
import numpy as np

def data_preprocess(df, x_seq_len = 1440, y_seq_len = 60, stride = 5, batch_size = 1008, idx = None):
    print('data processing')
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

    if idx is not None:
        print("looping index = " + str(idx))
    else:
        print('looping index = 0 ')
    x = None
    y = None
    new_idx = None
    loop_idx = idx if idx is not None else x_seq_len + y_seq_len     # the first index of loop

    for i in range(loop_idx ,result.shape[0],stride):
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
            x = np.concatenate([x, x_df.values[None,:,:]], axis=0)\
        
        if y is None:
            y = y_data[None,:]
        else:
            y = np.concatenate([y, y_data[None,:]], axis=0)

        if x.shape[0] >= batch_size:
            new_idx = i
            break
    
    print("new_idx = {0} ".format(new_idx))
    loop_cond = (new_idx < (df.shape[0] - 1))
    print("loop cond = {0} ".format(loop_cond ))
    return x,y,new_idx,loop_cond


def macd_preprocess():
    df = pd.DataFrame.from_csv('datasets/crypto_hist/{0}_{1}.csv'.format('XRPUSD','1m'))
    df = df[::-1]
    # ema = pd.DataFrame.ewma(df["close"], span=10, freq="m")
    df['ema10'] = df['close'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
    print(df)
    
# macd_preprocess()
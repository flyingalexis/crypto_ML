# store functions to calculate indicators 
# BUT JUST FOR TESTING, THIS FILE SHOULD BE IN THE DATA SYSTEM INSTEAD OF THE MODEL 

def ema (close , previous_ema ,n):
    alpha = 2.0/ (float(n) + 1.0)
    prior = alpha * float(close)
    latter = float(previous_ema) * (1 - alpha)
    return prior + latter

df = pd.DataFrame.from_csv('datasets/crypto_hist/{0}_{1}.csv'.format('XRPUSD','1m'))
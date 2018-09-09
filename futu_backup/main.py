import futuquant as ft
import utils.export as export
import utils.database as database
import matplotlib.pyplot as plt
import extractRising as trend_extraction
import pandas as pd

read_from_server = True
ret_data = None

if read_from_server:
  print('hello')
  # get data from server
  quote_ctx = ft.OpenQuoteContext(host="35.237.154.81", port=11111)
  # 800000 means HSI in futu
  # ret_status, ret_data = quote_ctx.get_history_kline('HK.800000', start='2007-06-25', end='2007-09-01',ktype='K_DAY')
  ret_status, ret_data = quote_ctx.get_history_kline('HK.800000', start='2018-05-01', end='2018-05-02',ktype='K_15M')
  export.export_csv_pandas(ret_data, 'test.csv')
  # ret_status, ret_data = quote_ctx.get_history_kline('HK.20703', start='2007-06-25', end='2018-07-01',ktype='K_DAY')
else:
  # get testing data by reading csv
  ret_data = pd.read_csv('test2.csv')

# the stuff that we want to do ! ~~~
fluc_periods = trend_extraction.get_flucuating_periods(ret_data)
# fig, ax = plt.subplots(1, 1)
# ret_data.plot(ax=ax,color='b')
plt.plot(ret_data.index, ret_data['close'], '-')
# for period in fluc_periods:
  # print(ret_data.index)
  # print(ret_data.index[period[0]:period[1]])
  #ret_data.iloc[period[0]:period[1]].plot(x='time_key',y='close')
  #plt.plot(ret_data.index[period[0]:period[1]], ret_data['close'], '-')
plt.show()

export.export_csv_pandas(ret_data, 'suisun')
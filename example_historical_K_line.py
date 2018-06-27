import futuquant as ft
import utils.export as export
import utils.database as database
import matplotlib.pyplot as plt

quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)

# 800000 means HSI in futu
ret_status, ret_data = quote_ctx.get_history_kline('HK.800000', start='2007-06-25', end='2007-07-01',ktype='K_DAY')

# plt.plot(ret_data['time_key'], ret_data['close'], '-')
# plt.show()
export.export_csv_pandas(ret_data, 'test2')

database.record_k_line(ret_data, 'K_day')

#print(ret_data)
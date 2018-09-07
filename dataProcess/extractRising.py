import utils.export as export
import numpy as np

# hard code threshold
MIN_Y_RANGE = 0.05
MIN_X_RANGE = 5
MAX_Y_FLUC = 0.01

def __get_fluctuating(df, trend, head = True) :
  cum_wave_changes = 0
  last_wave_index = trend[0]
  if head:
    targeting_df = df.iloc[trend[0] :trend[1]]
  else:
    targeting_df = df.iloc[trend[1] :trend[0]:-1]
  for index, row in targeting_df.iterrows():
    daily_change = 0
    if( not np.isnan(row['periodic_changes%'])):
      daily_change = float(row['periodic_changes%'])
    cum_wave_changes = cum_wave_changes + daily_change
    # calculate mean of change
    changing_mean = cum_wave_changes/((index - trend[0] + 2) if head else (abs(index - trend[1]) + 2))
    # make sure mean of change <= threshold and make sure abs(cum - mean of change) <= threshold
    if abs(abs(cum_wave_changes) - abs(changing_mean)) > MAX_Y_FLUC or abs(changing_mean) > MAX_Y_FLUC :
      fluc_period = ( trend[0] , index ) if head else (index, trend[1])
      return fluc_period

def get_flucuating_periods(df):
  # data preparation :
  # calculate data (periodic_changes%) which will be used for the trend extraction
  start_y = df.iloc[0]['close']
  df['periodic_changes%'] = (df['close'] - df['close'].shift()) / start_y

  # model the stock market into rising & decreasing trends
  # and exclude unstable changes (which we are not tartgeting)
  # trends output are (start_index, end_index, % of change during the period)
  trend_list = []
  starting_index = 0
  while starting_index < df.shape[0]:
    x_index,y_range = __extract_period_of_trend(df,starting_index)
    if x_index - starting_index >= MIN_X_RANGE and abs(y_range) >= MIN_Y_RANGE:
      trend_list.append((starting_index, x_index, y_range))
      starting_index = x_index
      continue
    starting_index = starting_index + 1
    
  # extract the changing period between the rising period and decreasing period
  print('trends : ')
  print(trend_list)

  flucuating_periods = []
  for trend in trend_list:
    flucuating_periods.append(__get_fluctuating(df, trend, True))
    flucuating_periods.append(__get_fluctuating(df, trend, False))

  print('flucating_periods before merge :')
  print(flucuating_periods)
  print(__merged_flucuating_periods(flucuating_periods))
  return __merged_flucuating_periods(flucuating_periods)

# def __extract_period_of_trend(df,start_x):
#   #find the largest cummulated change (end point)
#   maximum_wave_index = start_x
#   maximum_wave_changes = 0 
#   stashed_changes = 0
#   trend = None  # 1 -> rising , -1 -> decresing 
#   for index, row in df.iloc[start_x :].iterrows():
#     daily_change = 0
#     if( not np.isnan(row['periodic_changes%'])):
#       daily_change = float(row['periodic_changes%'])
#     sum_of_changes = maximum_wave_changes + daily_change + stashed_changes

#     # extract trend of starting point
#     if(trend is None and sum_of_changes != 0):
#       trend = np.where(sum_of_changes > 0, 1, -1)

#     # when the agent back to (starting_y) mean of max change , return the current max wave
#     elif( trend is not None and np.sign(sum_of_changes) != trend ):
#       return (maximum_wave_index, maximum_wave_changes)

#     if abs(sum_of_changes) >= abs(maximum_wave_changes):
#       maximum_wave_changes = sum_of_changes
#       maximum_wave_index = index
#       stashed_changes = 0
#     else:
#       stashed_changes = stashed_changes + float(daily_change)

#   return (maximum_wave_index, maximum_wave_changes)

def __extract_period_of_trend(df,start_x):
  #find the largest cummulated change (end point)
  maximum_wave_index = start_x
  maximum_wave_changes = 0 
  sum_of_changes = 0
  trend = None  # 1 -> rising , -1 -> decresing 
  start_price = df['close'].iloc[start_x]
  print('start_x')
  print(start_x)
  print(start_price)
  for index, row in df.iloc[start_x :].iterrows():
    daily_change = 0
    if( not np.isnan(row['periodic_changes%'])):
      daily_change = float(row['periodic_changes%'])
    sum_of_changes = daily_change + sum_of_changes

    if abs(sum_of_changes) >= abs(maximum_wave_changes):
      # print('new max !')
      maximum_wave_changes = sum_of_changes
      maximum_wave_index = index
    # extract trend of starting point
    if(trend is None and sum_of_changes != 0):
      trend = np.where(sum_of_changes > 0, 1, -1)
    # when the agent back to (starting_y) mean of max change , return the current max wave
    elif( trend is not None ):
      COND_DOMAIN_THRESHOLD = index - start_x > 10
      COND_AVAERAGE_THRESHOLD = abs(daily_change) < abs(maximum_wave_changes / maximum_wave_index - start_x + 1)
      COND_CHANGE_OF_TREND = np.sign(row['close'] - start_price) != trend
      if( COND_CHANGE_OF_TREND ):
        print(maximum_wave_index, maximum_wave_changes)
        return (maximum_wave_index, maximum_wave_changes)

  return (maximum_wave_index, maximum_wave_changes)


def __merged_flucuating_periods(flucuating_periods):
  merged_flucuating_periods = []
  merged = []
  for index,period in enumerate(flucuating_periods):
    if index not in merged:
      if len(flucuating_periods) - 1 == index:
        merged_flucuating_periods.append(period)
      elif period[1] + 1 == flucuating_periods[index + 1][0]:
        merged_flucuating_periods.append((period[0], flucuating_periods[index + 1][1]))
        merged.append(index + 1)
      else:
        merged_flucuating_periods.append(period)
  return merged_flucuating_periods
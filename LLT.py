import pymysql
import pandas as pd
import numpy as np
from ipdb import set_trace
import statsmodels.api as sm
from hurst import compute_Hc

from matplotlib.pyplot import plot as plt

from datetime import datetime, timedelta

# ----------------------------------------------------
# Documentation:
# Data: Extract from 2 spreadsheet: `ra_index` and `ra_nav` under DATABASE `mofang`;
#       Using `globalid` in sheet 1 and `ra_index_id` in sheet 2, match full information (sheet1)
#           and net-asset-value (sheet2) time series;
#       Create a DATAFRAME storing basic information of all index
#       For every index, assign a DATAFSERIES representing time series of n.a.v.;
# Index: 中证800指数, 标普高盛黄金人民币计价指数(Wind商品指数), 中证全债指数, 中证货币基金, 沪深300，中证货币基金指数
# globalid: 120000084, 120000080, 120000092, 120000039, 120000001, 120000039
# ----------------------------------------------------


conn_1 = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')

sql_container = {'index_id': "SELECT `globalid`, `ra_code`, `ra_name` FROM `ra_index`"}

info_ds = pd.read_sql(sql=sql_container['index_id'], con=conn_1, index_col=['globalid'])

# info_ds = info_ds.set_index(keys='globalid', drop=True)
# info_ds.index = info_ds.index.astype(str)

sql_index = "SELECT `ra_date`, `ra_nav` FROM `ra_index_nav` WHERE `ra_index_id`="

TS_dict = dict()
for item in list(info_ds.index):
    temp_sql = sql_index + str(item) + ";"
    TS_dict[item] = pd.read_sql(sql=temp_sql, con=conn_1, index_col=['ra_date'], parse_dates=['ra_date'])
# TS_dict:
# key: int,    value:DataFrame

# ----------------------------------------------------
# R-S analysis : revised
def divide_RS(time_series, n):
    # n: divide TS into n parts each containing at most 'ceil(len(time_series)/n) elements
    if len(time_series) < n:
        return
    else:
        length_per_ts = int(np.floor(len(time_series)/n))
        list_rescaled_range = []
        for i in range(n):
            if i == n-1:
                ts_temp = time_series.iloc[i*length_per_ts:]
                list_rescaled_range.append(rescaled_range_log(ts_temp))
            else:
                ts_temp = time_series.iloc[i*length_per_ts:(i+1)*length_per_ts]
                list_rescaled_range.append(rescaled_range_log(ts_temp))

        list_range = pd.Series(list_rescaled_range)
        return np.log(list_range.mean()), np.log(n)






# what if there are some missing values ?  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def rescaled_range_log(time_series):
    M = time_series.mean()
    ts_move = time_series.apply(lambda x: x-M)
    Z = ts_move.values.cumsum()
    R = Z.max() - Z.min()
    S = time_series.std()
    if S == 0:
        return
    else:
        return R/S
    


def hurst_index(time_series, sampling_info):
    # sampling_info = [begin, step_size, sampling_size]
    if sampling_info[2] <= 0:
        return
    else:
        # size of time_series would be about 5000.
        if sampling_info[0] + (sampling_info[2]-1) * sampling_info[1] > len(time_series) - 1:
            return
        else:
            list_y = []
            list_x = []
            for i in range(sampling_info[2]):
                y, x = divide_RS(time_series, sampling_info[0] + i * sampling_info[1])
                list_y.append(y)
                list_x.append(x)

            Y = np.array(list_y)
            X = np.array(list_x)

            set_trace()
            model = sm.OLS(Y,sm.add_constant(X))
            results = model.fit()
            return results.params[1]



# -----------------------------------------------------
# Index: 中证800指数，标普高盛黄金人民币计价指数(Wind商品指数)，中证全债指数，中证货币基金, 沪深300
# globalid: 120000084, 120000080 , 120000092, 120000039, 120000001
# core_variable: TS_dict



temp = TS_dict[120000084].iloc[:,0]
sampling_info  = [20, 100, 15]

L = hurst_index(temp, sampling_info)




# -----------------------------------------------------
# plot demo
# fig = plt.figure()
# ax_zhongzheng = fig.add_subplot(111)
# TS_dict[120000084].plot(ax=ax_zhongzheng)




# -----------------------------------------------------
# Simple Moving Average
def SMA(time_series, time_step):
    if time_step <= 1:
        return
    time_step = int(np.floor(time_step))
    res_ts = time_series.copy()
    for i in range(len(time_series))[time_step-1:]:
        res_ts.iloc[i] = time_series.iloc[i-time_step+1:i].sum()/time_step

    # res_ts.iloc[:time_step] = np.nan
    return res_ts

# Exponential Moving Average
def EMA(time_series, alpha):
    if alpha < 0 or alpha > 1:
        return
    else:
        res_ts = time_series.copy()
        for i in range(len(time_series))[1:]:
            res_ts.iloc[i] = alpha * time_series.iloc[i] + (1-alpha) * res_ts.iloc[i-1]
        # res_ts =
        return res_ts


def low_pass_filter(x, alpha):
    if x == 0:
        return

    numerator = np.power(1-alpha/2, 2) * (1-2/x+1/np.power(x,2))
    denominator = 1 - 2*(1-alpha)/x + np.power(1-alpha,2)/np.power(x,2)

    if denominator == 0:
        return
    return  numerator/denominator


def apply_low_pass_filter(time_series, alpha):
    if alpha < 0 or alpha > 1:
        return
    else:
        res_ts = time_series.copy()
        for key, value in res_ts.items():
            res_ts[key] = low_pass_filter(value, alpha)
        return res_ts




# ------------------------------------------
# 上证指数 demo: globalid to be searched

# to be continued




# ------------------------------------------

def adjust_ratio(list_ts, time):
    current_date = datetime.strptime(time, '%Y-%m-%d')

    d = 60
    alpha = 2/(1+d)
    list_strength = []
    for i in range(len(list_ts)):
        llt = apply_low_pass_filter(list_ts[0], alpha)
        # assume 'current_date' and before is valid for Series llt, denominator is nonzero. To be continued ...
        strength = (llt[current_date] - llt[current_date - timedelta(1)]) / llt[current_date - timedelta(1)]
        if strength < 0:
            list_strength.append(0)
        else:
            list_strength.append(strength)

    return list(np.array(list_strength) / np.array(list_strength).sum())






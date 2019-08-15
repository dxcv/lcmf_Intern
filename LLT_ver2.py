import pymysql
import pandas as pd
import numpy as np
from ipdb import set_trace
import statsmodels.api as sm
from hurst import compute_Hc

import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import time
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
# establish time_series pool with every index in dictionary 'TS_dict'

conn_1 = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')
sql_container = {'index_id': "SELECT `globalid`, `ra_code`, `ra_name` FROM `ra_index`"}
info_ds = pd.read_sql(sql=sql_container['index_id'], con=conn_1, index_col=['globalid'])
sql_index = "SELECT `ra_date`, `ra_nav` FROM `ra_index_nav` WHERE `ra_index_id`="

TS_dict = dict()
for item in list(info_ds.index):
    temp_sql = sql_index + str(item) + ";"
    TS_dict[item] = pd.read_sql(sql=temp_sql, con=conn_1, index_col=['ra_date'], parse_dates=['ra_date'])

# TS_dict:
# key: int,    value:DataFrame

# return time_series with sorted dates and 'nan' dropped
def ts_preprocessing(time_series):
    time_series = time_series.sort_index()


    if len(time_series == 0) > 0:
        time_series[time_series == 0] = np.nan
        time_series = time_series.fillna(method='ffill')
        time_series = time_series.fillna(method='bfill')

    return time_series.dropna()

# return list of time_series according to key from dictionary
def get_from_dict(ts_dict, key):
    return ts_preprocessing(ts_dict[key].iloc[:,0])



# -----------------------------------------------------
# Simple Moving Average
def SMA(time_series, time_step):
    if time_step <= 1:
        return
    time_step = int(np.floor(time_step))
    res_ts = time_series.copy()
    for i in range(len(time_series))[time_step - 1:]:
        res_ts.iloc[i] = time_series.iloc[i - time_step + 1:i].sum() / time_step

    # res_ts.iloc[:time_step] = np.nan
    return res_ts


# Exponential Moving Average
def EMA(time_series, alpha):
    if alpha < 0 or alpha > 1:
        return
    else:
        res_ts = time_series.copy()
        for i in range(len(time_series))[1:]:
            res_ts.iloc[i] = alpha * time_series.iloc[i] + (1 - alpha) * res_ts.iloc[i - 1]
        # res_ts =
        return res_ts

# return LLT time_series
def apply_low_pass_filter(time_series, alpha):

    ini = time_series.iloc[0]
    if alpha < 0 or alpha > 1:
        return
    else:
        res_ts = time_series.copy()
        for i in range(len(res_ts)):
            if i < 2:
                res_ts.iloc[i] = np.power(alpha,2)*time_series.iloc[i] + 2*(1-alpha)*ini - np.power(1-alpha,2)*ini
            else:
                res_ts.iloc[i] = np.power(alpha,2) * time_series.iloc[i] + 2*(1-alpha)*res_ts.iloc[i-1] - np.power(1-alpha,2)*res_ts.iloc[i-2]
        return res_ts


# ------------------------------------------
# return a list of ratio
def list_llt(list_ts, alpha):
    # d = 10
    # alpha = 2 / (1 + d)

    res = []

    for item1 in list_ts:
        res.append(apply_low_pass_filter(item1, alpha))
    return res


def adjust_ratio(llt, adjustment_time):
    current_date = datetime.strptime(adjustment_time, '%Y-%m-%d')

    d = 10
    alpha = 2 / (1 + d)
    list_strength = []

    for item1 in llt:
        strength = (item1[current_date] - item1[current_date - timedelta(1)]) / item1[current_date - timedelta(1)]
        if strength < 0:
            list_strength.append(0)
        else:
            list_strength.append(strength)

    return list(np.array(list_strength) / np.array(list_strength).sum())


# ------------------------------------------
# authentification
# Index: 中证800指数，标普高盛黄金人民币计价指数(Wind商品指数)，中证全债指数，中证货币基金, 沪深300, 上证指数
# globalid: 120000084, 120000080 , 120000092, 120000039, 120000001, 120000016

# return a pd.Series presenting Hurst index information
def ds_hurst(list_time_series, list_columns):
    ds = pd.Series({
    list_columns[0]: compute_Hc(list_time_series[0])[0],
    list_columns[1]: compute_Hc(list_time_series[1])[0],
    list_columns[2]: compute_Hc(list_time_series[2])[0],
    list_columns[3]: compute_Hc(list_time_series[3])[0],
})

    return ds

# return a list of time_series from a dictionary
def extract_source_nav(time_series_dict, list_globalid):
    L = []
    for item1 in list_globalid:
        L.append(get_from_dict(time_series_dict, item1))
    return L

# show or save a image of time_series in list
def trend_plot(list_time_series, list_columns=('list_name1','list_name2'), plot_index=('title','xlabel','ylabel'), path = ''):
    L = list_time_series

    if len(L) < 2:
        inner_join_dates = L[0].index
    else:
        # intersection of all time-series date index
        inner_join_dates = L[0].index & L[1].index
        if len(L) >= 3:
            for i in range(len(L))[2:]:
                inner_join_dates = inner_join_dates & L[i].index

    add_columns = dict()
    for i in range(len(list_columns)):
        key = list_columns[i]
        add_columns[key] = L[i].loc[inner_join_dates]

    df = pd.DataFrame(add_columns)

    fig = plt.figure(figsize=(10,5))
    ax_plot = fig.add_subplot(111)
    df.plot(ax=ax_plot)
    # plot_index = [title, xlabel, ylabel]
    ax_plot.set_title(plot_index[0])
    ax_plot.set_xlabel(plot_index[1])
    ax_plot.set_ylabel(plot_index[2])

    ax_plot.legend(list_columns, loc='best')
    plt.rcParams['font.sans-serif']=['SimHei']

    if path == '':
        plt.show()
    else:
        figure_name = path + plot_index[0] + '.png'
        plt.savefig(figure_name, dpi=400, bbox_inches='tight')

    return

# return a list of time_series with aligned time index
def align(list_time_series):
    L = list_time_series

    inner_join_dates = L[0].index & L[1].index

    if len(L) >= 3:
        for i in range(len(L))[2:]:
            inner_join_dates = inner_join_dates & L[i].index
    return list(map(lambda x: x.loc[inner_join_dates], L))



# Part 1:
# 四种指数的净值对比
# l1 = [120000084, 120000080 , 120000092, 120000039]
# l2 = ['中证800指数','标普高盛黄金人民币计价指数','中证全债指数','中证货币基金']
# l3 = ['过去十三年四种指数走势图', '日期', '指数净值']
#
# ds_hurst(extract_source_nav(TS_dict, l1), l2)
#
# trend_plot(extract_source_nav(TS_dict, l1), l2, l3)


# Part 2:
# 上证指数与EMA对比
# L = extract_source_nav(TS_dict, [120000016])
# d = 60
# L.append(EMA(L[0], 2/(d+1)))
#
# # 短序列中移动平均的平滑性显示明显
# L[0] = L[0].loc['2015-12-15':'2016-04-08']
# L[1] = L[1].loc['2015-12-15':'2016-04-08']
#
# trend_plot(L, ['上证指数', 'EMA'], ['上证指数净值与EMA对比', '日期', '指数净值'])


# Part 3:
# 上阵指数与LLT对比
# L1 = extract_source_nav(TS_dict, [120000016])
# d = 5
# L1.append(apply_low_pass_filter(L1[0], 2/(d+1)))
#
# L1[0] = L1[0].loc['2015-12-15':'2016-04-08']
# L1[1] = L1[1].loc['2015-12-15':'2016-04-08']
# trend_plot(L1, ['上证指数', 'LLT'], ['上证指数净值与LLT对比', '日期', '指数净值'])


# Part 4:
# adjustment
l1 = [120000084, 120000080 , 120000092, 120000039]
l2 = ['中证800指数','标普高盛黄金人民币计价指数','中证全债指数','中证货币基金']
list_ts_four = extract_source_nav(TS_dict, l1)
list_ts_four = align(list_ts_four)
list_ts_four = [ts_preprocessing(item) for item in list_ts_four]
# adjust_ratio(list_ts_four, '2005-02-01')



# print(datetime(2018,2,1) + relativedelta(months=1))

def scale_func(ds):
    return ds.apply(lambda x: x/ds.iloc[0])

def scale_func_cumprod(ds):
    temp = ds.pct_change()
    temp.iloc[0] = 0
    temp = (temp + 1).cumprod()
    return temp

def scale_func_log_return(ds):
    temp = ds.pct_change()
    temp = temp + 1
    temp.apply(lambda x: np.log(x))
    return temp


def MaximumDrawDown(ts):
    res = 0
    ts_max = ts.iloc[0]

    if ts_max <= 0:
        return

    for i in range(len(ts)):
        if ts.iloc[i] > ts_max:
            ts_max = ts.iloc[i]

        temp = (ts_max - ts.iloc[i])/ts_max

        if temp > res:
            res = temp

    return res





def TAA(list_time_series, list_columns, time_info, alpha):
    temp = dict()
    for i in range(len(list_time_series)):
        temp[list_columns[i]] = list_time_series[i]
    df = pd.DataFrame(temp)
    # df = df.apply(scale_func_cumprod, axis=0)

    df  = df.apply(lambda x: x.pct_change(), axis=0)
    df = df.fillna(value=0)

    res = pd.Series({'TAA':[]}, index=df.index)

    # time_info = [begin_date]
    begin_date = datetime.strptime(time_info[0], '%Y-%m-%d')
    delta = relativedelta(months=1)

    # nav time_series to compute ratio of adjustment
    llt = list_llt(list_time_series, alpha)

    # initialization
    adjust_date = begin_date + delta
    ratio_vector = np.array([1/len(list_time_series)]*len(list_time_series))

    for i in range(len(res)):
        current_date = res.index[i]
        res.iloc[i] = df.loc[current_date].values.dot(ratio_vector)

        if adjust_date == current_date:
            ratio_vector = np.array(adjust_ratio(llt, current_date.strftime("%Y-%m-%d")))
            adjust_date = current_date + delta

    res = (res+1).cumprod()
    return res



# time series from TAA model
res = TAA(list_ts_four, ['中证800指数','标普高盛黄金人民币计价指数','中证全债指数','中证货币基金'], ['2005-01-04'], 2/61)

zhongzheng = scale_func_cumprod(list_ts_four[0])
trend_plot([zhongzheng, res], ['中证800指数', 'TAA'], ['中证800指数净值与TAA对比', '日期', '指数比'])

gold = scale_func_cumprod(list_ts_four[1])
trend_plot([gold, res],['标普高盛黄金人民币计价指数', 'TAA'], ['标普高盛黄金人民币计价指数净值与TAA对比', '日期', '指数比'] )


bond = scale_func_cumprod(list_ts_four[2])
trend_plot([bond, res],['中证全债指数', 'TAA'], ['中证全债指数与TAA对比', '日期', '指数比'] )

money = scale_func_cumprod(list_ts_four[3])
trend_plot([money, res], ['中证货币基金', 'TAA'], ['中证货币基金与TAA对比', '日期', '指数比'] )

hushen = scale_func_cumprod(extract_source_nav(TS_dict, [120000001])[0])
trend_plot([hushen, res], ['沪深300', 'TAA'], ['沪深300与TAA对比', '日期', '指数比'] )


trend_plot([res['2006-01-25':'2015-01-25'], hushen['2006-01-25':'2015-01-25']], ['TAA', '沪深300'], ['沪深300与TAA对比', '日期', '指数比'] )



trend_plot([list_ts_four[1]['2006-06-06':'2015-09-08']], ['标普高盛黄金人民币计价指数'], ['标普高盛黄金人民币计价指数', '日期', '指数值'])


H = ds_hurst(list_ts_four,['中证800指数','标普高盛黄金人民币计价指数','中证全债指数','中证货币基金'])

llt = list_llt(list_ts_four, 2/61)
trend_plot(llt, ['中证800指数','标普高盛黄金人民币计价指数','中证全债指数','中证货币基金'], ['四种指数的LLT','日期','指数'])

trend_plot(list_ts_four, ['中证800指数','标普高盛黄金人民币计价指数','中证全债指数','中证货币基金'], ['四种指数','日期','指数'])
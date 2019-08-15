import pymysql
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from ipdb import set_trace


# input: 原始时间序列
# output: 按 日期 排序 并 向前（后）填充缺失值的时间序列
def ts_preprocess(ts):
    tmp = ts.sort_index()
    tmp = tmp.fillna(method='ffill')
    tmp = tmp.fillna(method='bfill')
    tmp = tmp.reindex(tradeDays.index)
    return tmp


# 对 时间序列 list 进行 日期交集 批处理
def list_ts_preprocess(list_ts):
    list_tmp = [ts_preprocess(ts) for ts in list_ts]

    common_index = list_tmp[0].index & list_tmp[1].index
    if len(list_tmp) > 2:
        for ts in list_tmp[2:]:
            common_index = common_index & ts.index

    list_res = [ts.reindex(common_index) for ts in list_tmp]

    return list_res


# input: 交易所缩写
# output: 交易日期时间序列
def TradeDay_EXCHMARKET():
    pass


# 数据库连接，sql注入
#conn_factor = pymysql.connect(host='192.168.88.12', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')
#sql_bond = 'SELECT `TRADE_DT`, `S_DQ_CLOSE` FROM `cbondindexeodcnbd` WHERE `S_INFO_WINDCODE` = '

conn_hpc = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')
sql_hpc = "SELECT * FROM `trade_dates`;"




def myplot(ts):
    fig = plt.figure(figsize=(40, 10))
    ax = fig.add_subplot(111)
    df = pd.DataFrame({'ra_nav':ts})
    df.plot(ax=ax)
    plt.show()


# input: 按‘日期’排序的 存量 时间序列
# output: 累积收益率 时间序列
def culmulReturn(ts, log=False):
    if log:
        # 对数收益率
        return (ts.pct_change().fillna(0)+1).apply(lambda x: np.log(x)).cumsum()
    else:
        return (ts.pct_change().fillna(0)+1).cumprod()






data_path = 'E:/1-Career/lcmf/factorModel'

# Part 1: 交易日数据：上海证券交易所
tradeDays = pd.read_excel(data_path+'/mofang_TRADE_DAYS.xls', sheet_name='全部交易日', parse_dates=['td_date'])
tradeDays.set_index('td_date', inplace=True)
tradeDays.sort_index(inplace=True)
tradeDays.index.name = 'TRADE_DAYS'

# Part 2: 自变量--CRBRI工业现货指数，Shibor:3个月，信用利差AAA，美元指数
x_df = pd.read_excel(data_path+'/信用利差(中位数)产业债不同评级.xls',
                     skiprows=[0,1,2],
                     names=['日期', '信用利差(中位数):产业债AAA', 'SHIBOR:3个月'])
x_df.set_index('日期', inplace=True)
x_df = x_df.reindex(tradeDays.index)


USDIndex_df = pd.read_excel(data_path+'/USDIndex.xls', parse_dates=['日期'])
USDIndex_df = USDIndex_df[['日期', '收盘价(元)']]
USDIndex_df.rename(columns={
    '日期': 'TRADE_DAYS',
    '收盘价(元)': 'USD美元指数'
}, inplace=True)
USDIndex_df.set_index('TRADE_DAYS', inplace=True)
USDIndex_df.index.name = x_df.index.name

CRBRI_df = pd.read_excel(data_path+'/CRBRI工业现货指数.xls', parse_dates=['日期'])
CRBRI_df = CRBRI_df[['日期', '收盘价(元)']]
CRBRI_df.rename(columns={
    '日期': 'TRADE_DAYS',
    '收盘价(元)': 'CRBRI工业现货指数'
}, inplace=True)
CRBRI_df.set_index('TRADE_DAYS', inplace=True)
CRBRI_df.index.name = x_df.index.name

x_df = x_df.merge(USDIndex_df, how='left', on='TRADE_DAYS', sort=True)
x_df = x_df.merge(CRBRI_df, how='left', on='TRADE_DAYS', sort=True)
x_df.dropna(inplace=True)

continuousPeriod = tradeDays[x_df.index[0]: x_df.index[-1]]
x_df = x_df.reindex(continuousPeriod.index).fillna(method='ffill')


# Part 3: 因变量--中证500,
y_df = pd.DataFrame(None, index=tradeDays.index)

zhongzheng500_df = pd.read_excel(data_path+'/中证500全收益.xls', parse_dates=['TRADE_DT'])
zhongzheng500_df = zhongzheng500_df[['TRADE_DT', 'S_DQ_CLOSE']]
zhongzheng500_df.rename(columns={
    'TRADE_DT': 'TRADE_DAYS',
    'S_DQ_CLOSE': '中证500全收益'
}, inplace=True)
zhongzheng500_df = zhongzheng500_df.set_index('TRADE_DAYS').sort_index()
y_df = y_df.merge(zhongzheng500_df, how='left', on='TRADE_DAYS', sort=True)

hushen300_df = pd.read_excel(data_path+'/沪深300全收益.xls', parse_dates=['TRADE_DT'])
hushen300_df = hushen300_df[['TRADE_DT', 'S_DQ_CLOSE']]
hushen300_df.rename(columns={
    'TRADE_DT': 'TRADE_DAYS',
    'S_DQ_CLOSE': '沪深300全收益'
}, inplace=True)
hushen300_df = hushen300_df.set_index('TRADE_DAYS').sort_index()
y_df = y_df.merge(hushen300_df, how='left', on='TRADE_DAYS', sort=True)

creditBonds_df = pd.read_excel(data_path+'/中债-企业债AAA财富(总值)指数.xls', parse_dates=['日期'])
creditBonds_df = creditBonds_df[['日期', '收盘价(元)']]
creditBonds_df.rename(columns={
    '日期': 'TRADE_DAYS',
    '收盘价(元)': '信用债AAA'
}, inplace=True)
creditBonds_df = creditBonds_df.set_index('TRADE_DAYS').sort_index()
y_df = y_df.merge(creditBonds_df, how='left', on='TRADE_DAYS', sort=True)

rateShortBonds_df = pd.read_excel(data_path+'/中债-3-5年期国债财富(总值)指数.xls', parse_dates=['日期'])
rateShortBonds_df = rateShortBonds_df[['日期', '收盘价(元)']]
rateShortBonds_df.rename(columns={
    '日期': 'TRADE_DAYS',
    '收盘价(元)': '短期利率债'
}, inplace=True)
rateShortBonds_df = rateShortBonds_df.set_index('TRADE_DAYS').sort_index()
y_df = y_df.merge(rateShortBonds_df, how='left', on='TRADE_DAYS', sort=True)

rateLongBonds_df = pd.read_excel(data_path+'/中债-7-10年期国债财富(总值)指数.xls', parse_dates=['日期'])
rateLongBonds_df = rateLongBonds_df[['日期', '收盘价(元)']]
rateLongBonds_df.rename(columns={
    '日期': 'TRADE_DAYS',
    '收盘价(元)': '长期利率债'
}, inplace=True)
rateLongBonds_df = rateLongBonds_df.set_index('TRADE_DAYS').sort_index()
y_df = y_df.merge(rateLongBonds_df, how='left', on='TRADE_DAYS', sort=True)

gold_df = pd.read_excel(data_path+'/黄金指数.xls', parse_dates=['日期'])
gold_df = gold_df[['日期', '收盘价(元)']]
gold_df.rename(columns={
    '日期': 'TRADE_DAYS',
    '收盘价(元)': '黄金指数'
}, inplace=True)
gold_df = gold_df.set_index('TRADE_DAYS').sort_index()
y_df = y_df.merge(gold_df, how='left', on='TRADE_DAYS', sort=True)

y_df.dropna(inplace=True)


# Part 4: x y 数据对齐，转化为累积收益率序列
y_align_df = y_df.reindex(x_df.index).fillna(method='ffill')
y_return_df = y_align_df.apply(culmulReturn, axis=0)

with pd.ExcelWriter(data_path+'/CoreData.xls') as ExcelWriter:
    x_df.to_excel(ExcelWriter, sheet_name='x_df')
    y_return_df.to_excel(ExcelWriter, sheet_name='y_return_df')
    ExcelWriter.save()












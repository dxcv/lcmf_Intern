import pymysql
import pandas as pd
import numpy as np
from packages_ct.fun_lib import *
from matplotlib import pyplot as plt

from ipdb import set_trace



# asset classes
conn_asset = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')
sql_container = {'index_id': "SELECT `globalid`, `ra_code`, `ra_name` FROM `ra_index`"}
info_ds = pd.read_sql(sql=sql_container['index_id'], con=conn_asset, index_col=['globalid'])
sql_index = "SELECT `ra_date`, `ra_nav` FROM `ra_index_nav` WHERE `ra_index_id`="

TS_dict = dict()
for item in list(info_ds.index):
    temp_sql = sql_index + str(item) + ";"
    TS_dict[item] = pd.read_sql(sql=temp_sql, con=conn_asset, index_col=['ra_date'], parse_dates=['ra_date'])


trans = {
    '上证50指数': 120000051,
    '中小板指数': 120000036,
    '创业板指数': 120000018,
    '中证800指数': 120000084,
    '中证国债指数': 120000010,
    '中证信用债指数': 120000011,
    '中证可转债指数': 120000012,
    '标普高盛黄金指数': 120000044,
    '南华商品指数': 120000029,
    '中证货币基金指数': 120000039
}
trans_verse = dict(zip(trans.values(), trans.keys()))
list_ts_asset = []

for key in trans_verse.keys():
    list_ts_asset.append(TS_dict[key])


# input: a time_series
# output: a new time_series with no missing value and sorted date index
def ts_preprocess(ts):
    tmp = ts.sort_index()
    tmp = tmp.fillna(method='ffill')
    tmp = tmp.fillna(method='bfill')
    return tmp

def list_ts_preprocess(list_ts):
    list_tmp = [ts_preprocess(ts) for ts in list_ts]

    common_index = list_tmp[0].index & list_tmp[1].index
    if len(list_tmp) > 2:
        for ts in list_tmp[2:]:
            common_index = common_index & ts.index

    list_res = [ts.reindex(common_index) for ts in list_tmp]

    return list_res


list_ts_asset = list_ts_preprocess(list_ts_asset)





# factors
conn_factor = pymysql.connect(host='192.168.88.12', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')
sql_bond = 'SELECT `TRADE_DT`, `S_DQ_CLOSE` FROM `cbondindexeodcnbd` WHERE `S_INFO_WINDCODE` = '
TS_dict_factor = dict()
TS_dict_factor[120000016] = pd.read_sql(sql=sql_index+'120000016;', con=conn_asset, index_col=['ra_date'], parse_dates=['ra_date'])
TS_dict_factor[120000017] = pd.read_sql(sql=sql_index+'120000017;', con=conn_asset, index_col=['ra_date'], parse_dates=['ra_date'])

TS_dict_factor['CBA04601.CS'] = pd.read_sql(sql=sql_bond+"'CBA04601.CS';", con=conn_factor, index_col=['TRADE_DT'], parse_dates=['TRADE_DT'])
TS_dict_factor['CBA06501.CS'] = pd.read_sql(sql=sql_bond+"'CBA06501.CS';", con=conn_factor, index_col=['TRADE_DT'], parse_dates=['TRADE_DT'])

trans_factor = {
    '上证指数': 120000016,
    '深证成指': 120000017,
    '中债3-5年期国债指数': 'CBA04601.CS',
    '中债7-10年期国债指数': 'CBA06501.CS'
}
trans_verse_factor = dict(zip(trans_factor.values(), trans_factor.keys()))


proxiesEG_df = pd.DataFrame(
    {
        '上证指数': TS_dict_factor[120000016]['ra_nav'],
        '深证成指': TS_dict_factor[120000017]['ra_nav']
    }
)

proxiesEG_df['上证指数涨跌幅'] = proxiesEG_df['上证指数'].pct_change()
proxiesEG_df['深证成指涨跌幅'] = proxiesEG_df['深证成指'].pct_change()
proxiesEG_ts = proxiesEG_df.apply(lambda x: x['上证指数涨跌幅']*0.5 + x['深证成指涨跌幅']*0.5, axis=1)
proxiesEG_ts = (proxiesEG_ts.sort_index().fillna(0)+1).cumprod()


def myplot(ts):
    fig = plt.figure(figsize=(40, 10))
    ax = fig.add_subplot(111)
    df = pd.DataFrame({'ra_nav':ts})
    df.plot(ax=ax)
    plt.show()

proxiesRR_ts = (TS_dict_factor['CBA04601.CS']['S_DQ_CLOSE'].sort_index().pct_change().fillna(0)+1).cumprod()

import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from packages_ct import os_ct

from ipdb import set_trace




def loading(path):
    '''
    策略表
    :param path:
    :return:
    '''
    dfSource = pd.read_csv(path, index_col='TRADE_DT', parse_dates=['TRADE_DT'])
    return dfSource


def DBconnection(DBname):
    '''
    Choose 'DBname' between: wind, mofang
    '''
    if DBname == 'wind':
        conn = pymysql.connect(host='192.168.88.12', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')
    else:
        conn = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')
    return conn


current_path = 'E:/1-Career/lcmf/dataCore.csv'
def mainModel():
    dfCore = loading(current_path)

    conn = DBconnection('wind')
    # 沪深300全收益
    sqlMy1 = "SELECT TRADE_DT, S_DQ_CLOSE FROM aindexeodprices WHERE S_INFO_WINDCODE = 'H00300.CSI';"
    sqlMy2 = "SELECT TRADE_DT, S_DQ_CLOSE FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA02001.CS'"

    hushen300_df = pd.read_sql(sql=sqlMy1, con=conn, parse_dates=['TRADE_DT'])
    hushen300_df.set_index('TRADE_DT', inplace=True)
    hushen300_df.sort_index(inplace=True)

    bond_df = pd.read_sql(sql=sqlMy2, con=conn, parse_dates=['TRADE_DT'])
    bond_df.set_index('TRADE_DT', inplace=True)
    bond_df.sort_index(inplace=True)




    assets_df = hushen300_df.merge(bond_df, how='inner', on='TRADE_DT', sort=True)

    datesIndex = assets_df.index & dfCore.index

    assets_df = assets_df.reindex(datesIndex)
    assets_df.rename(columns={'S_DQ_CLOSE_x':'hushen300', 'S_DQ_CLOSE_y': 'bond'}, inplace=True)

    dfCore = dfCore.reindex(datesIndex)

    assetsPCT_df = assets_df.pct_change()
    assetsPCT_df = assetsPCT_df.fillna(0)



    # Part 1: TTM
    assets_TTM = assetsPCT_df.copy()
    assets_TTM['compBool_TTM'] = dfCore['compBool_TTM']
    assets_TTM['selection'] = pd.Series()

    for index in assets_TTM.index:
        row = assets_TTM.loc[index]

        if row['compBool_TTM'] == 1:
            row['selection'] = row['hushen300']
        else:
            row['selection'] = row['bond']

        assets_TTM.loc[index] = row

    ts_TTM = (assets_TTM['selection'] + 1).cumprod()

    # Part 2: exAnte
    assets_exAnte = assetsPCT_df.copy()
    assets_exAnte['compBool_exAnte'] = dfCore['compBool_exAnte']
    assets_exAnte['selection'] = pd.Series()

    for index in assets_exAnte.index:
        row = assets_exAnte.loc[index]

        if row['compBool_exAnte'] == 1:
            row['selection'] = row['hushen300']
        else:
            row['selection'] = row['bond']

        assets_exAnte.loc[index] = row

    ts_exAnte = (assets_exAnte['selection'] + 1).cumprod()

    res_df = pd.DataFrame({'TTM': ts_TTM, 'exAnte': ts_exAnte})
    # res_df = pd.DataFrame({'TTM': assets_TTM['selection'], 'exAnte': assets_exAnte['selection']})
    return (assetsPCT_df+1).cumprod(), res_df



def reverse():
    dftmp = loading(current_path)
    dfNew = dftmp[['E_P_TTM', 'E_P_exAnte']]
    dfChange = dfNew.rolling(window=3).apply(lambda x: x.iloc[1] - x.iloc[0], raw=False)
    dfChange = dfChange.fillna(0)

    def judge(x):
        if x > 0:
            return 0
        else:
            return 1

    dfChange['compBool_TTM'] = dfChange['E_P_TTM'].apply(judge)
    dfChange['compBool_exAnte'] = dfChange['E_P_exAnte'].apply(judge)


    dfCore = dfChange.copy()
    conn = DBconnection('wind')
    # 沪深300全收益
    sqlMy1 = "SELECT TRADE_DT, S_DQ_CLOSE FROM aindexeodprices WHERE S_INFO_WINDCODE = 'H00300.CSI';"
    sqlMy2 = "SELECT TRADE_DT, S_DQ_CLOSE FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA02001.CS'"

    hushen300_df = pd.read_sql(sql=sqlMy1, con=conn, parse_dates=['TRADE_DT'])
    hushen300_df.set_index('TRADE_DT', inplace=True)
    hushen300_df.sort_index(inplace=True)

    bond_df = pd.read_sql(sql=sqlMy2, con=conn, parse_dates=['TRADE_DT'])
    bond_df.set_index('TRADE_DT', inplace=True)
    bond_df.sort_index(inplace=True)

    assets_df = hushen300_df.merge(bond_df, how='inner', on='TRADE_DT', sort=True)

    datesIndex = assets_df.index & dfCore.index

    assets_df = assets_df.reindex(datesIndex)
    assets_df.rename(columns={'S_DQ_CLOSE_x': 'hushen300', 'S_DQ_CLOSE_y': 'bond'}, inplace=True)

    dfCore = dfCore.reindex(datesIndex)



    assetsPCT_df = assets_df.pct_change()
    assetsPCT_df = assetsPCT_df.fillna(0)

    # Part 1: TTM
    assets_TTM = assetsPCT_df.copy()
    assets_TTM['compBool_TTM'] = dfCore['compBool_TTM']
    assets_TTM['selection'] = pd.Series()

    for index in assets_TTM.index:
        row = assets_TTM.loc[index]

        if row['compBool_TTM'] == 1:
            row['selection'] = row['hushen300']
        else:
            row['selection'] = row['bond']

        assets_TTM.loc[index] = row

    ts_TTM = (assets_TTM['selection'] + 1).cumprod()

    # Part 2: exAnte
    assets_exAnte = assetsPCT_df.copy()
    assets_exAnte['compBool_exAnte'] = dfCore['compBool_exAnte']
    assets_exAnte['selection'] = pd.Series()

    for index in assets_exAnte.index:
        row = assets_exAnte.loc[index]

        if row['compBool_exAnte'] == 1:
            row['selection'] = row['hushen300']
        else:
            row['selection'] = row['bond']

        assets_exAnte.loc[index] = row

    ts_exAnte = (assets_exAnte['selection'] + 1).cumprod()

    res_df = pd.DataFrame({'TTM': ts_TTM, 'exAnte': ts_exAnte})
    # res_df = pd.DataFrame({'TTM': assets_TTM['selection'], 'exAnte': assets_exAnte['selection']})
    return dfChange, (assetsPCT_df + 1).cumprod(), res_df



# nav_df, strategy_df = mainModel()
reverseStrategy, nav_df, strategy_df = reverse()

path = 'E:/1-Career/lcmf/res_reverse.xls'
ExcelWriter = pd.ExcelWriter(path)
reverseStrategy.to_excel(ExcelWriter, sheet_name='strategyLogic')
nav_df.to_excel(ExcelWriter, sheet_name='nav')
strategy_df.to_excel(ExcelWriter, sheet_name='strategy')
ExcelWriter.save()
ExcelWriter.close()


















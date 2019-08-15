import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse

from ipdb import set_trace

def DBconnection(DBname):
    '''
    Choose 'DBname' between: wind, mofang
    '''
    if DBname == 'wind':
        conn = pymysql.connect(host='192.168.88.12', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')
    else:
        conn = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')
    return conn


def datetimeProcess(date):
    if type(date) == str:
        return parse(date)
    else:
        return date


def hushen300StockCodeSet(date):
    '''
    return set of codes of stocks at date: e.x. '2019-08-05'
    '''
    # Sheet 3: aindexmembers  key: S_INFO_WINDCODE, S_CON_INDATE, S_CON_OUTDATE
    conn = DBconnection(DBname='wind')
    sqlMy = "SELECT S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE FROM `aindexmembers` WHERE `S_INFO_WINDCODE` = '000300.SH';"
    dfSource = pd.read_sql(sql=sqlMy, con=conn, parse_dates=['S_CON_INDATE', 'S_CON_OUTDATE'])
    currentDate = datetimeProcess(date)
    mask1 = (dfSource['S_CON_OUTDATE'].isnull()) & (dfSource['S_CON_INDATE'] <= currentDate)
    mask2 = ((dfSource[~mask1])['S_CON_INDATE'] <= currentDate) & (currentDate <= (dfSource[~mask1])['S_CON_OUTDATE'])
    dfFilter = dfSource[mask1 | mask2]
    if len(dfFilter) == 300:
        return list(dfFilter['S_CON_WINDCODE'].values)
    else:
        print('Selected %d stocks;' % len(dfFilter))
        return list(dfFilter['S_CON_WINDCODE'].values)
    







# TTM data:
# Sheet 1: ashareconsensusrollingdata  key: S_INFO_WINDCODE, EST_DT, ROLLING_TYPE, NET_PROFIT
# Sheet 2: ashareeodderivativeindicator  key: S_INFO_WINDCODE, TRADE_DT, S_VAL_MV, S_VAL_PE_TTM, FLOAT_A_SHR_TODAY, S_DQ_CLOSE_TODAY    ''' Please note that S_VAL_MV and FLOAT_A_SHR_TODAY has to multiply 10000

def EP(date):
    '''
    
    '''

    conn = DBconnection(DBname='wind')
    sqlMy1 = "SELECT S_INFO_WINDCODE, NET_PROFIT FROM ashareconsensusrollingdata WHERE ROLLING_TYPE = 'FTTM' AND EST_DT = '" + str(date) + "';"
    dfSource = pd.read_sql(sql=sqlMy1, con=conn)

    if len(dfSource) == 0:
        return -1, -1

    StockCodeSet = hushen300StockCodeSet(date)  # list
    mask = dfSource['S_INFO_WINDCODE'].apply(lambda x: x in StockCodeSet)
    NetProfit = dfSource[mask]  # dataframe
    NetProfit.set_index('S_INFO_WINDCODE', inplace=True)

    sqlMy2 = "SELECT S_INFO_WINDCODE, S_VAL_MV, S_VAL_PE_TTM, FLOAT_A_SHR_TODAY, S_DQ_CLOSE_TODAY FROM ashareeodderivativeindicator WHERE TRADE_DT =  '" + str(date) + "';" 
    dfSourceValue = pd.read_sql(sql=sqlMy2, con=conn)
    maskValue = dfSourceValue['S_INFO_WINDCODE'].apply(lambda x: x in StockCodeSet)
    Value = dfSourceValue[maskValue]
    Value.set_index('S_INFO_WINDCODE', inplace=True)
    Value['S_VAL_PE_TTM'].fillna(0)

    Value['S_VAL_MV'] = Value['S_VAL_MV'].multiply(10000)
    Value['FLOAT_A_SHR_TODAY'] = Value['FLOAT_A_SHR_TODAY'].multiply(10000)

    NetProfit['S_VAL_MV'] = Value['S_VAL_MV']
    NetProfit['E_P'] = NetProfit.apply(lambda x: x.iloc[0] / x.iloc[1], axis=1)

    Value['E_P'] = Value['S_VAL_PE_TTM'].apply(lambda x: 1/x)
    Value['FLOAT_VAL_MV'] = Value[['FLOAT_A_SHR_TODAY', 'S_DQ_CLOSE_TODAY']].apply(lambda x: x.iloc[0] * x.iloc[1], axis=1)

    dfCore = pd.DataFrame(
        {
            'TTM': NetProfit['E_P'],
            'ex_ante': Value['E_P'],
            'FLOAT_VAL_MV': Value['FLOAT_VAL_MV']
        }
    )

    sumVAL_MV = dfCore['FLOAT_VAL_MV'].sum()
    dfCore['WEIGHT'] = dfCore['FLOAT_VAL_MV'].apply(lambda x: x/sumVAL_MV)
    res_TTM = (dfCore['TTM'] * dfCore['WEIGHT']).sum()
    res_exAnte = (dfCore['ex_ante'] * dfCore['WEIGHT']).sum()

    return res_TTM, res_exAnte



def bondYTM():
    '''
    中债企业债总指数 S_INFO_WINDCOED CBA02001.CS
    '''
    # date = datetimeProcess(date)

    conn = DBconnection('wind')
    sqlMy = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA02001.CS';"
    dfSource = pd.read_sql(sql=sqlMy, con=conn, parse_dates=['TRADE_DT'])

    dfSource.set_index('TRADE_DT', inplace=True)
    dfSource.sort_index(inplace=True)

    return dfSource


def logicTimingModel(riskPremium = 0):
    dfSource = bondYTM()     # 1 connection
    dfSource.reset_index(inplace=True)
    dfSource['AVG_CASH_YTM'] = dfSource['AVG_CASH_YTM'].multiply(0.01)
    dfSource['E_P_TTM'] = pd.Series()
    dfSource['E_P_exAnte'] = pd.Series()
    dfSource['compBool_TTM'] = pd.Series()
    dfSource['compBool_exAnte'] = pd.Series()

    df = dfSource.copy()
    for index in df.index:
        row = df.loc[index]

        date = pd.Timestamp.to_pydatetime(row['TRADE_DT'])
        date = date.strftime("%Y-%m-%d")
        date = date.replace('-', '')

        A, B = EP(date)
        row['E_P_TTM'] = A
        row['E_P_exAnte'] = B
        if row['E_P_TTM'] >= row['AVG_CASH_YTM'] + riskPremium:
            # stock
            row['compBool_TTM'] = 1
        else:
            # bond
            row['compBool_TTM'] = 0

        if row['E_P_exAnte'] >= row['AVG_CASH_YTM'] + riskPremium:
            # stock
            row['compBool_exAnte'] = 1
        else:
            # bond
            row['compBool_exAnte'] = 0

        df.loc[index] = row


    return df












# if __name__ == '__main__':
# experimentalDate = '20100805'
# codeSet = hushen300StockCodeSet(experimentalDate)
# if codeSet:
#     print([code for code in codeSet[:5]])

# for date in pd.date_range('20100101', periods=100):
#     # A, B = TTM(date)
#     # print("date: %s -- A: %d, B: %d; " % date.strftime("%Y-%m-%d-%H-%M-%S") % len(A) % len(B) )
#     L = hushen300StockCodeSet(date)
#     print("date: %s -- num: %d; " %(date.strftime("%Y-%m-%d"), len(L)))

# A, B = TTM(experimentalDate)

# df = bondYTM()

#
#
#
# datetime.weekday(parse('2006-11-20'))

# date = '2019-07-25'
# date = date.replace('-', '')
# res = EP(date)


# df = logicTimingModel()

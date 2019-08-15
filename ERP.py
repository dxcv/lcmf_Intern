#coding=utf8

import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Simhei'] #解决中文显示问题，目前只知道黑体可行
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题
from statsmodels.formula.api import ols
import statsmodels.api as sm


from ipdb import set_trace

def DBconnection(DBname):
    '''
    Choose 'DBname' between: wind, mofang
    '''
    if DBname == 'wind':
        conn = pymysql.connect(host='192.168.88.11', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')
    else:
        conn = pymysql.connect(host='192.168.88.254', user='root', passwd='Mofang123', database='mofang', charset='utf8')
    return conn
def datetimeProcess(date):
    if type(date) == str:
        return parse(date)
    else:
        return date
def timestampTostr(date):
    return (str(date)[:10]).replace('-', '')

# 2.1 基于DDM模型计算的风险溢价
class ERP2_1:
    def __init__(self):
        self.conn = DBconnection('wind')



    def hushen300(self):
        '''
        hushen300_df: ['DIVIDEND_YIELD', 'PE_TTM', 'MV_FLOAT', 'DIVIDEND_TOTAL', 'DIVIDEND_YIELD_PAYMENT_RATE']
        hushen300quan_df: ['S_DQ_CLOSE', 'FUTURE_1_Y_RETURN']

        :return:
        '''
        sqlMy = "SELECT TRADE_DT, DIVIDEND_YIELD, PE_TTM, MV_FLOAT, EST_PE_Y1, TOT_SHR_FLOAT FROM aindexvaluation WHERE S_INFO_WINDCODE = '000300.SH';"
        hushen300_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        hushen300_df.sort_index(inplace=True)
    
        hushen300_df['DIVIDEND_TOTAL'] = hushen300_df['DIVIDEND_YIELD'] * hushen300_df['MV_FLOAT'] / 100
        hushen300_df['DIVIDEND_YIELD_PAYMENT_RATE'] = hushen300_df['DIVIDEND_YIELD'] * hushen300_df['PE_TTM'] / 100
    
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM `aindexeodprices` WHERE `S_INFO_WINDCODE` = 'H00300.CSI';"
        hushen300quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        hushen300quan_df.sort_index(inplace=True)

        hushen300quan_df['FUTURE_1_Y_RETURN'] = pd.Series()
        for index in hushen300quan_df.index:
            future = index + timedelta(days=250)
            if future in hushen300quan_df.index:
                hushen300quan_df.loc[index, 'FUTURE_1_Y_RETURN'] = hushen300quan_df.loc[future, 'S_DQ_CLOSE'] / \
                                                                   hushen300quan_df.loc[index, 'S_DQ_CLOSE'] - 1
        hushen300quan_df['FUTURE_1_Y_RETURN'].fillna(method='ffill', inplace=True)

        return hushen300_df, hushen300quan_df
    
    # 10 CBA04501    3-5 CBA04601
    def bond(self):
        '''
        bond3_5_df: YTM
        bond10_df: YTM
        :return:
        '''

        sqlMy1 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA04501.CS';"
        sqlMy2 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA04601.CS';"
        bond10_df = pd.read_sql(sql=sqlMy1, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        bond3_5_df = pd.read_sql(sql=sqlMy2, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        bond10_df.sort_index(inplace=True)
        bond3_5_df.sort_index(inplace=True)
        return bond3_5_df, bond10_df

    @staticmethod
    def inflation():
        '''

        :return: CPI_PPI_df, GDPpingjian_df
        '''
        path = "E:/1-Career/lcmf/Projects/ERP"
        # CPI = pd.read_excel(path + "/CPI.xls", parse_dates=['date'])
        # PPI = pd.read_excel(path + "/PPI.xls", parse_dates=['date'])
        #
        # dfCore = pd.merge(left=CPI, right=PPI, how='inner', left_on='date', right_on='date')
        # dfCore.set_index('date', inplace=True)
        # dfCore['Inflation'] = dfCore.apply(lambda x: (x.iloc[0] + x.iloc[1]) / 2, axis=1)

        dfCore = pd.read_excel(path + "/inflation.xls", parse_dates=['date'])
        dfCore.set_index('date', inplace=True)


        return dfCore[['CPI:当月同比(月)', 'PPI:全部工业品:当月同比(月)']], dfCore[['GDP:平减指数:GDP:累计同比(季)']]

    # @staticmethod
    # Simple Moving Average
    # def SMA(time_series, time_step):
    #     if time_step <= 1:
    #         return
    #     time_step = int(np.floor(time_step))
    #     res_ts = time_series.copy()
    #     for i in range(len(time_series))[time_step - 1:]:
    #         res_ts.iloc[i] = time_series.iloc[i - time_step + 1:i].sum() / time_step
    #
    #     # res_ts.iloc[:time_step] = np.nan
    #     return res_ts

    @staticmethod
    def SMA(time_series, time_step):
        return time_series.rolling(window=time_step).mean()




    # 2.1.1 股息率减无风险利率 (FED MODEL)

    # Ok
    def align_nominal_2111(self):
        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, bond10_df = self.bond()
        dateInterval = (hushen300_df.index & bond10_df.index) & (bond3_5_df.index & hushen300quan_df.index)
    
        hushen300_df = hushen300_df.reindex(dateInterval)
        bond3_5_df = bond3_5_df.reindex(dateInterval)
        bond10_df = bond10_df.reindex(dateInterval)
        hushen300quan_df = hushen300quan_df.reindex(dateInterval)

    
        figure1_data = pd.DataFrame(None, columns=['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM', 'hushen300Future1YReturn'], index=dateInterval)
        figure1_data['hushen300DividendRate-bond3_5YTM'] = hushen300_df['DIVIDEND_YIELD'] - bond3_5_df['AVG_CASH_YTM']
        figure1_data['hushen300DividendRate-bond10YTM'] = hushen300_df['DIVIDEND_YIELD'] - bond10_df['AVG_CASH_YTM']
        figure1_data['hushen300Future1YReturn'] = hushen300quan_df['FUTURE_1_Y_RETURN'] * 100
    
        figure2_data = pd.DataFrame(None, columns=['DividendTotal', 'DividendPaymentRate'])
        figure2_data['DividendTotal'] = hushen300_df['DIVIDEND_TOTAL']
        figure2_data['DividendPaymentRate'] = hushen300_df['DIVIDEND_YIELD_PAYMENT_RATE']
    
    
        return figure1_data, figure2_data

    # Ok
    def figure1Plot_2111(self):
        figure1_data, figure2_data = self.align_nominal_2111()
        # fig = plt.figure(figsize=(8,6))
        # axMy = fig.add_subplot(111)
        # axMy2 = axMy.twinx()
        #
        # figure1_data.plot(ax=axMy)
        # axMy.set_title('FED Model')
        # axMy.legend(['沪深300股息率-10Y国债', '沪深300股息率-1Y国债', '未来1年股票投资回报(rhs)'], loc='best')
        # axMy.set_xlabel('时间')
        #
        # plt.rcParams['font.sans-serif']=['SimHei']
        # # plt.savefig('risk3.png', dpi=400, bbox_inches='tight')
        # plt.show()

        # figure1_data.astype(dtype=float)
        # figure1_data[['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM']].plot()
        # figure1_data['hushen300Future1YReturn'].plot(secondary_y=True)
        # ax = figure1_data.plot(secondary_y=['hushen300Future1YReturn'])
        # plt.show()

        fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2)
        ax1.plot(figure1_data.index, figure1_data.iloc[:,0].values, 'r-', figure1_data.index, figure1_data.iloc[:,1].values, 'b.')
        # ax1.legend()
        ax1.set_title('股息率减无风险利率')
        ax1.set_ylabel('%')

        ax3.plot(figure1_data.index, figure1_data.iloc[:,2].values, 'g')
        ax3.set_title('未来一年收益(rhs)')
        ax3.set_ylabel('%')

        ax2.plot(figure2_data.index, figure2_data['DividendTotal']/100000000)
        ax2.set_title('现金分红总额')
        ax2.set_ylabel('亿元')

        ax4.plot(figure2_data.index, figure2_data['DividendPaymentRate'])
        ax4.set_title('股息支付率')
        ax4.set_ylabel('%')

        plt.show()

    # Ok
    def align_real_2111(self):
        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, bond10_df = self.bond()
        dateInterval = (hushen300_df.index & bond10_df.index) & (bond3_5_df.index & hushen300quan_df.index)

        hushen300_df = hushen300_df.reindex(dateInterval)
        bond3_5_df = bond3_5_df.reindex(dateInterval)
        bond10_df = bond10_df.reindex(dateInterval)
        hushen300quan_df = hushen300quan_df.reindex(dateInterval)

        hushen300quan_df['FUTURE_1_Y_RETURN'] = pd.Series()
        for index in hushen300quan_df.index:
            future = index + timedelta(days=250)
            if future in hushen300quan_df.index:
                hushen300quan_df.loc[index, 'FUTURE_1_Y_RETURN'] = hushen300quan_df.loc[future, 'S_DQ_CLOSE'] / \
                                                                   hushen300quan_df.loc[index, 'S_DQ_CLOSE'] - 1
        hushen300quan_df['FUTURE_1_Y_RETURN'].fillna(method='ffill', inplace=True)


        # 2016 - 2018
        # inflation_df = self.inflation()
        # inflation_df = inflation_df.reindex(dateInterval)
        # inflation_df.fillna(method='ffill', inplace=True)
        # inflation_df.fillna(method='bfill', inplace=True)


        CPI_PPI_df, GDPpingjian_df = self.inflation()
        CPI_PPI_df = CPI_PPI_df.reindex(dateInterval)
        CPI_PPI_df.fillna(method='ffill', inplace=True)
        CPI_PPI_df.fillna(method='bfill', inplace=True)

        GDPpingjian_df = GDPpingjian_df.reindex(dateInterval)
        GDPpingjian_df.fillna(method='ffill', inplace=True)
        GDPpingjian_df.fillna(method='bfill', inplace=True)

        inflation_df = CPI_PPI_df.copy()
        inflation_df['Inflation'] = (inflation_df['CPI:当月同比(月)'] + inflation_df['PPI:全部工业品:当月同比(月)']) / 2

        figure1_data = pd.DataFrame(None, columns=['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM',
                                                   'hushen300Future1YReturn'], index=dateInterval)
        figure1_data['hushen300DividendRate-bond3_5YTM'] = hushen300_df['DIVIDEND_YIELD'] - bond3_5_df['AVG_CASH_YTM'] + inflation_df['Inflation']
        figure1_data['hushen300DividendRate-bond10YTM'] = hushen300_df['DIVIDEND_YIELD'] - bond10_df['AVG_CASH_YTM'] + inflation_df['Inflation']
        figure1_data['hushen300Future1YReturn'] = hushen300quan_df['FUTURE_1_Y_RETURN'] * 100


        figure2_data = pd.DataFrame({
            'GDP:平减指数:GDP:累计同比(季)' : GDPpingjian_df['GDP:平减指数:GDP:累计同比(季)'],
            'Inflation' : inflation_df['Inflation']
        }, index=inflation_df.index)

        return figure1_data, figure2_data

    # Ok
    def figure12Plot_2112(self):
        figure1_data, figure2_data = self.align_real_2111()

        figure1_data.astype(dtype=float)
        figure1_data[['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM']].plot()
        figure1_data['hushen300Future1YReturn'].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['hushen300Future1YReturn'])

        ax2 = figure2_data.plot()
        plt.show()
        # fig, (ax1, ax2) = plt.subplot(1,2)
        # figure1_data.plot(ax=ax1, figsize=(8,6), title='股息率减无风险利率', secondary_y=figure1_data['hushen300Future1YReturn'])
        # ax2.bar(x=figure2_data.index, y=figure2_data['DividendTotal'])
        # ax2.plot(x=figure2_data.index, y=figure2_data['DividendPaymentRate'])
        #
        # plt.legend()
        # plt.show()



        # fig = plt.figure(figsize=(8, 6))
        # axMy = fig.add_subplot(111)
        #
        # figure1_data.plot(ax=axMy)
        # axMy.set_title('FED Model')
        # axMy.legend(['沪深300股息率-10Y国债真实利率', '沪深300股息率-1Y国债真实利率', '未来1年股票投资回报(rhs)'], loc='best')
        # axMy.set_xlabel('时间')
        #
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # # plt.savefig('risk3.png', dpi=400, bbox_inches='tight')
        # plt.show()





    # 2.1.2 用市盈率的倒数减国债收益率 名义/实际
    # PE_TTM: 静态市盈率
    # Ok
    def align_TTM_2121(self, real=False):
        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, _ = self.bond()

        dateInterval = (hushen300_df.index & hushen300quan_df.index) & bond3_5_df.index

        hushen300_df = hushen300_df.reindex(dateInterval)
        bond3_5_df = bond3_5_df.reindex(dateInterval)
        hushen300quan_df = hushen300quan_df.reindex(dateInterval)

        figure1_data = pd.DataFrame(None, columns=['ERP', 'ERP_SMA', '+2SD', '-2SD', 'Future1YReturn'],
                                    index=dateInterval)
        if real:
            CPI_PPI_df, GDPpingjian_df = self.inflation()
            CPI_PPI_df = CPI_PPI_df.reindex(dateInterval)
            CPI_PPI_df.fillna(method='ffill', inplace=True)
            CPI_PPI_df.fillna(method='bfill', inplace=True)

            inflation_df = CPI_PPI_df.copy()
            inflation_df['Inflation'] = (inflation_df['CPI:当月同比(月)'] + inflation_df['PPI:全部工业品:当月同比(月)']) / 2

            figure1_data['ERP'] = (1 / hushen300_df['PE_TTM']) - ( (bond3_5_df['AVG_CASH_YTM'] - inflation_df['Inflation']) / 100 )
        else:
            figure1_data['ERP'] = (1 / hushen300_df['PE_TTM']) - ( bond3_5_df['AVG_CASH_YTM'] / 100)

        figure1_data['ERP_SMA'] = self.SMA(figure1_data['ERP'], 250)
        # std = figure1_data['ERP_SMA'].std()
        figure1_data['+2SD'] = figure1_data['ERP_SMA'] + 2 * figure1_data['ERP_SMA'].rolling(window=100).std()
        figure1_data['-2SD'] = figure1_data['ERP_SMA'] - 2 * figure1_data['ERP_SMA'].rolling(window=100).std()
        figure1_data['Future1YReturn'] = hushen300quan_df['FUTURE_1_Y_RETURN']

        return figure1_data

    # Ok
    def figure1Plot_2121(self):
        figure1_data = self.align_TTM_2121(real=False)

        # 条件截取
        figure1_data = figure1_data[figure1_data.index > '20100101']

        figure1_data.astype(dtype=float)

        figure1_data[['ERP', 'ERP_SMA', '+2SD', '-2SD']].plot()
        figure1_data[['Future1YReturn']].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['Future1YReturn'])

        plt.show()

    # EST_PE_Y1: 动态市盈率
    # Ok
    def align_EST_PE_2122(self):
        '''
        figure1_data: ['ERP_FutureEP_nominal', 'ERP_FutureEP_real', 'Future1YReturn']
        :return:
        '''
        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, _ = self.bond()
        inflation_df = self.inflation()

        dateInterval = (hushen300_df.index & hushen300quan_df.index) & bond3_5_df.index

        hushen300_df = hushen300_df.reindex(dateInterval)
        bond3_5_df = bond3_5_df.reindex(dateInterval)
        hushen300quan_df = hushen300quan_df.reindex(dateInterval)

        # inflation_df = inflation_df.reindex(dateInterval)
        # inflation_df.fillna(method='ffill', inplace=True)
        # inflation_df.fillna(method='bfill', inplace=True)

        CPI_PPI_df, GDPpingjian_df = self.inflation()
        CPI_PPI_df = CPI_PPI_df.reindex(dateInterval)
        CPI_PPI_df.fillna(method='ffill', inplace=True)
        CPI_PPI_df.fillna(method='bfill', inplace=True)

        inflation_df = CPI_PPI_df.copy()
        inflation_df['Inflation'] = (inflation_df['CPI:当月同比(月)'] + inflation_df['PPI:全部工业品:当月同比(月)']) / 2

        figure1_data = pd.DataFrame(None, columns=['ERP_FutureEP_nominal', 'ERP_FutureEP_real', 'Future1YReturn'],
                                    index=dateInterval)
        figure1_data['ERP_FutureEP_nominal'] = (1 / hushen300_df['EST_PE_Y1']) - ( bond3_5_df['AVG_CASH_YTM'] / 100 )
        figure1_data['ERP_FutureEP_real'] = (1 / hushen300_df['EST_PE_Y1']) - ( ( bond3_5_df['AVG_CASH_YTM'] - inflation_df['Inflation'] ) / 100 )
        figure1_data['Future1YReturn'] = hushen300quan_df['FUTURE_1_Y_RETURN']

        return figure1_data

    # Ok
    def figure12Plot_2122(self):
        figure1_data = self.align_EST_PE_2122()

        # 条件截取
        figure1_data = figure1_data[figure1_data.index > '20130101']

        figure1_data.astype(dtype=float)

        # ['ERP_FutureEP_nominal', 'ERP_FutureEP_real', 'Future1YReturn']
        figure1_data[['ERP_FutureEP_nominal']].plot()
        figure1_data[['Future1YReturn']].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['Future1YReturn'])

        plt.show()

    # CAPE: cyclically adjusted PE
    # Ok
    def align_CAPE_2123(self):
        '''
        hushen300_df.columns=
        ['DIVIDEND_YIELD', 'PE_TTM', 'MV_FLOAT', 'EST_PE_Y1', 'PE_LYR',
       'DIVIDEND_TOTAL', 'DIVIDEND_YIELD_PAYMENT_RATE']
        :return:
        '''
        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, bond10_df = self.bond()
        dateInterval = (hushen300_df.index & bond10_df.index) & (bond3_5_df.index & hushen300quan_df.index)

        hushen300_df = hushen300_df.reindex(dateInterval)
        bond3_5_df = bond3_5_df.reindex(dateInterval)
        bond10_df = bond10_df.reindex(dateInterval)
        hushen300quan_df = hushen300quan_df.reindex(dateInterval)

        # inflation_df = self.inflation()
        # inflation_df = inflation_df.reindex(dateInterval)
        # inflation_df.fillna(method='ffill', inplace=True)
        # inflation_df.fillna(method='bfill', inplace=True)

        CPI_PPI_df, GDPpingjian_df = self.inflation()
        CPI_PPI_df = CPI_PPI_df.reindex(dateInterval)
        CPI_PPI_df.fillna(method='ffill', inplace=True)
        CPI_PPI_df.fillna(method='bfill', inplace=True)

        inflation_df = CPI_PPI_df.copy()
        inflation_df['Inflation'] = (inflation_df['CPI:当月同比(月)'] + inflation_df['PPI:全部工业品:当月同比(月)']) / 2


        # PE_TTM
        # hushen300_df['CAPE'] = hushen300_df['PE_TTM'] / (1 + inflation_df['Inflation'] / 100)

        CAPE_df = pd.DataFrame(None, columns=['S_DQ_CLOSE', 'earnings', 'Inflation'], index=dateInterval)
        CAPE_df['S_DQ_CLOSE'] = hushen300quan_df['S_DQ_CLOSE']
        CAPE_df['earnings'] = hushen300_df['MV_FLOAT'] / hushen300_df['PE_TTM']
        CAPE_df['Inflation'] = inflation_df['Inflation']

        rollingYear = 1
        CAPE_df['CAPE'] = (CAPE_df['S_DQ_CLOSE'] / CAPE_df['Inflation'] * hushen300_df['TOT_SHR_FLOAT']) / ( (CAPE_df['earnings'] / CAPE_df['Inflation']).rolling(window=rollingYear*250).mean() )

        figure1_data = pd.DataFrame(None, columns=['CAPE10Y', 'CAPE3_5Y', 'Future1YReturn'], index=dateInterval)
        figure1_data['CAPE10Y'] = (1 / CAPE_df['CAPE']) - (bond10_df['AVG_CASH_YTM'] / 100)
        figure1_data['CAPE3_5Y'] = (1 / CAPE_df['CAPE']) - (bond3_5_df['AVG_CASH_YTM'] / 100)
        figure1_data['Future1YReturn'] = hushen300quan_df['FUTURE_1_Y_RETURN']

        return figure1_data

    # Ok
    def figure1Plot_2123(self):
        figure1_data = self.align_CAPE_2123()

        # 条件截取
        figure1_data = figure1_data[figure1_data.index > '20050101']

        figure1_data.astype(dtype=float)

        # ['ERP_FutureEP_nominal', 'ERP_FutureEP_real', 'Future1YReturn']
        figure1_data[['CAPE10Y', 'CAPE3_5Y']].plot()
        figure1_data[['Future1YReturn']].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['Future1YReturn'])

        plt.show()








    # 2.1.3 两阶段及四阶段DDM公式
    # data not good ...
    def DDM2_2131(self):
        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, _ = self.bond()

        growth_df = pd.DataFrame(None, columns=['g_s_earnings', 'g_s_PE', 'g_l'], index=hushen300_df.index)
        growth_df['PE_TTM'] = hushen300_df['PE_TTM']
        growth_df['MV_FLOAT'] = hushen300_df['MV_FLOAT']
        growth_df['earnings'] = growth_df['MV_FLOAT'] / growth_df['PE_TTM']

        start = growth_df.index[0]
        for index in growth_df.index:
            delta = timedelta(days=250 * 5)
            if (index - delta) in growth_df.index:
                growth_df.loc[index, 'g_s_earnings'] = (growth_df.loc[index, 'earnings'] - growth_df.loc[
                    index - delta, 'earnings']) / growth_df.loc[index - delta, 'earnings']
            growth_df.loc[index, 'g_l'] = (growth_df.loc[index, 'earnings'] - growth_df.loc[start, 'earnings']) / \
                                          growth_df.loc[start, 'earnings']


        # g_s = 0.68 - 0.01*g_s_earnings^(-1) - 0.048*PE_TTM
        growth_df['g_s'] = 0.68 - 0.01 * (1 / growth_df['g_s_earnings']) - 0.048 * growth_df['PE_TTM']

        # g_s, g_l 单位为 1


        dateInterval = (growth_df.index & bond3_5_df.index) & hushen300quan_df.index


        bond3_5_df = bond3_5_df.reindex(dateInterval)
        hushen300quan_df = hushen300quan_df.reindex(dateInterval)
        growth_df = growth_df.reindex(dateInterval)



        H = 5
        D0 = hushen300_df.loc[dateInterval[0], 'DIVIDEND_TOTAL']
        idealPrice = pd.DataFrame({'g_s': growth_df['g_s'], 'g_l': growth_df['g_l']}, index=dateInterval)
        idealPrice.fillna(method='ffill', inplace=True)
        idealPrice.fillna(method='bfill', inplace=True)
        numerator = (1 + idealPrice['g_l']) + H * (idealPrice['g_s'] - idealPrice['g_l'])
        idealPrice['idealNav'] = (D0 * numerator) / (bond3_5_df['AVG_CASH_YTM']/100 - idealPrice['g_l'])

        # 有负值，待解决
        idealPrice['ERP'] = idealPrice['idealNav'].pct_change() - (bond3_5_df['AVG_CASH_YTM']/100)

        figure1data = pd.DataFrame({'ERP': idealPrice['ERP'], 'FUTURE_1Y_RETURN': hushen300quan_df['FUTURE_1_Y_RETURN']},
                                   index=dateInterval)
        return figure1data

    def figure1Plot_2131(self):
        figure1_data = self.DDM2_2131()

        # 条件截取
        figure1_data = figure1_data[figure1_data.index > '20090101']

        figure1_data.astype(dtype=float)

        # ['ERP_FutureEP_nominal', 'ERP_FutureEP_real', 'Future1YReturn']
        figure1_data[['ERP']].plot()
        figure1_data[['FUTURE_1Y_RETURN']].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['FUTURE_1Y_RETURN'])

        plt.show()

    # Ok
    def DDM4_2132(self):
        # EST_YOYPROFIT_Y1 预测净利润同比增速(Y1)
        sqlMy = "SELECT TRADE_DT, EST_YOYPROFIT_Y1, EST_YOYPROFIT_Y2 FROM aindexvaluation WHERE S_INFO_WINDCODE = '000300.SH';"
        YOYPROFIT_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'])
        YOYPROFIT_df.set_index('TRADE_DT', inplace=True)
        YOYPROFIT_df.sort_index(inplace=True)
        YOYPROFIT_df = YOYPROFIT_df.apply(lambda x: x / 100, axis=0)
        # 预测净利润同比增速(Y3) = [ 预测净利润同比增速(Y1) + 预测净利润同比增速(Y2) ] / 2
        YOYPROFIT_df['EST_YOYPROFIT_Y3'] = (YOYPROFIT_df['EST_YOYPROFIT_Y1'] + YOYPROFIT_df['EST_YOYPROFIT_Y2']) / 2
        # 单位为 1

        YOYPROFIT_df = YOYPROFIT_df.rename(
            columns={'EST_YOYPROFIT_Y1': 'g1', 'EST_YOYPROFIT_Y2': 'g2', 'EST_YOYPROFIT_Y3': 'g3'})
        YOYPROFIT_df['g_l'] = pd.Series(None)
        # 长期增速设为常数 0.01，如果大于无风险利率，term4 级数不收敛
        YOYPROFIT_df.fillna(0.01, inplace=True)

        hushen300_df, hushen300quan_df = self.hushen300()
        bond3_5_df, _ = self.bond()

        dateInterval = (hushen300_df.index & hushen300quan_df.index) & (YOYPROFIT_df.index & bond3_5_df.index)

        hushen300quan_df = hushen300quan_df.reindex(dateInterval)
        bond3_5_df = bond3_5_df.reindex(dateInterval)
        YOYPROFIT_df = YOYPROFIT_df.reindex(dateInterval)

        DE0 = hushen300_df.loc[dateInterval[0], 'DIVIDEND_YIELD_PAYMENT_RATE']
        bond3_5_df['AVG_CASH_YTM'] = bond3_5_df['AVG_CASH_YTM']/ 100

        figure1data = pd.DataFrame(columns=['ERP', 'FUTURE_1Y_RETURN'], index=dateInterval)
        figure1data['FUTURE_1Y_RETURN'] = hushen300quan_df['FUTURE_1_Y_RETURN']
        term1 = DE0 * (1 + YOYPROFIT_df['g1']) / (1 + bond3_5_df['AVG_CASH_YTM'])
        term2 = term1 * (1 + YOYPROFIT_df['g2']) / (1 + bond3_5_df['AVG_CASH_YTM'])
        term3 = term2 * (1 + YOYPROFIT_df['g3']) / (1 + bond3_5_df['AVG_CASH_YTM'])
        # 等比数列求和
        commonRatio = (1 + YOYPROFIT_df['g_l']) / (1 + bond3_5_df['AVG_CASH_YTM'])
        term4 = term3 * commonRatio / (1 - commonRatio)
        figure1data['ERP'] = 1 / (term1 + term2 + term3 + term4)

        return figure1data

    # Ok
    def figure1Plot_2132(self):
        figure1_data = self.DDM4_2132()

        # 条件截取
        figure1_data = figure1_data[figure1_data.index > '20150101']

        figure1_data.astype(dtype=float)

        # ['ERP_FutureEP_nominal', 'ERP_FutureEP_real', 'Future1YReturn']
        figure1_data[['ERP']].plot()
        figure1_data[['FUTURE_1Y_RETURN']].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['FUTURE_1Y_RETURN'])

        plt.show()



# 2.2 历史股权风险溢价
class ERP2_2:
    def __init__(self):
        self.conn = DBconnection('wind')

    def hushen300quan(self):
        '''
        hushen300_df: ['DIVIDEND_YIELD', 'PE_TTM', 'MV_FLOAT', 'DIVIDEND_TOTAL', 'DIVIDEND_YIELD_PAYMENT_RATE']
        hushen300quan_df: ['S_DQ_CLOSE', 'FUTURE_1_Y_RETURN']

        :return:
        '''

        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM `aindexeodprices` WHERE `S_INFO_WINDCODE` = 'H00300.CSI';"
        hushen300quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        hushen300quan_df.sort_index(inplace=True)

        hushen300quan_df['FUTURE_1_Y_RETURN'] = pd.Series()
        for index in hushen300quan_df.index:
            future = index + timedelta(days=250)
            if future in hushen300quan_df.index:
                hushen300quan_df.loc[index, 'FUTURE_1_Y_RETURN'] = hushen300quan_df.loc[future, 'S_DQ_CLOSE'] / \
                                                                   hushen300quan_df.loc[index, 'S_DQ_CLOSE'] - 1
        hushen300quan_df['FUTURE_1_Y_RETURN'].fillna(method='ffill', inplace=True)

        return hushen300quan_df

    # data not good
    def history(self):
        '''
        hushen300quan_df: ['S_DQ_CLOSE', 'FUTURE_1_Y_RETURN', 'Return', 'LongAVE', '5YAVE']
        :return:
        '''
        hushen300quan_df = self.hushen300quan()
        hushen300quan_df['Return'] = (hushen300quan_df['S_DQ_CLOSE'].pct_change()).fillna(0)
        hushen300quan_df['LongAVE'] = pd.Series()
        hushen300quan_df['5YAVE'] = pd.Series()

        start = hushen300quan_df.index[0]
        ts = hushen300quan_df['Return']
        for index in hushen300quan_df.index:
            hushen300quan_df.loc[index, 'LongAVE'] = ts[start:index].mean()

            delta = timedelta(days=250 * 5)
            if index - delta in hushen300quan_df.index:
                hushen300quan_df.loc[index, '5YAVE'] = ts[index - delta: index].mean()
        hushen300quan_df['5YAVE'].fillna(method='ffill', inplace=True)

        return hushen300quan_df


    def figure1Plot_22(self):
        figure1_data = self.history()
        figure1_data = figure1_data[['FUTURE_1_Y_RETURN', 'LongAVE', '5YAVE']]

        # 条件截取
        figure1_data = figure1_data[figure1_data.index > '20100101']

        figure1_data.astype(dtype=float)

        figure1_data[[ 'LongAVE', '5YAVE']].plot()
        figure1_data[['FUTURE_1_Y_RETURN']].plot(secondary_y=True)
        ax1 = figure1_data.plot(secondary_y=['FUTURE_1_Y_RETURN'])
        # ax1 = figure1_data.plot()
        plt.show()



# 2.3 截面回归与股权风险溢价
class ERP2_3:
    def __init__(self):
        self.conn = DBconnection('wind')

    @staticmethod
    def OLS(data_df):
        x = data_df.iloc[:, 0].values
        y = data_df.iloc[:, 1].values
        model = sm.OLS(y, sm.add_constant(x)).fit()
        return model

    def zhongzheng800StockCodeSet(self, date):
        '''
        中证800  000906.SH
        return set of codes of stocks at date: e.x. '2019-08-05'
        '''
        sqlMy = "SELECT S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE FROM aindexmembers WHERE S_INFO_WINDCODE = '000906.SH';"
        dfSource = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['S_CON_INDATE', 'S_CON_OUTDATE'])
        currentDate = parse(date)
        mask1 = (dfSource['S_CON_OUTDATE'].isnull()) & (dfSource['S_CON_INDATE'] <= currentDate)
        mask2 = ((dfSource[~mask1])['S_CON_INDATE'] <= currentDate) & (
                    currentDate <= (dfSource[~mask1])['S_CON_OUTDATE'])
        dfFilter = dfSource[mask1 | mask2]

        CodeSet = list(dfFilter['S_CON_WINDCODE'].values)
        return CodeSet

    def bond(self):
        '''
        bond3_5_df: YTM
        bond10_df: YTM
        :return:
        '''

        sqlMy1 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA04501.CS';"
        sqlMy2 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA04601.CS';"
        bond10_df = pd.read_sql(sql=sqlMy1, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        bond3_5_df = pd.read_sql(sql=sqlMy2, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        bond10_df.sort_index(inplace=True)
        bond3_5_df.sort_index(inplace=True)
        return bond3_5_df, bond10_df

    def timeRegression_1Stock(self, CodeStock, date, frequency=1):
        '''
        中证800全收益: H00906.CSI
        频率: X 个月
        :return:
        '''
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM aindexeodprices WHERE S_INFO_WINDCODE = 'H00906.CSI';"
        zhongzheng800quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'])
        zhongzheng800quan_df.set_index('TRADE_DT', inplace=True)
        zhongzheng800quan_df.sort_index(inplace=True)

        zhongzheng800quan_df['ROR'] = zhongzheng800quan_df['S_DQ_CLOSE'].pct_change()
        zhongzheng800quan_df.fillna(0, inplace=True)

        riskFree_df, _ = self.bond()

        sqlStock = "SELECT TRADE_DT, S_DQ_CLOSE FROM ashareeodprices WHERE S_INFO_WINDCODE = '" + str(CodeStock) + "';"
        Stock_df = pd.read_sql(sql=sqlStock, con=self.conn, parse_dates=['TRADE_DT'])
        Stock_df.set_index('TRADE_DT', inplace=True)
        Stock_df.sort_index(inplace=True)

        dateInterval = (riskFree_df.index & Stock_df.index) & zhongzheng800quan_df.index

        Stock_df = Stock_df.reindex(dateInterval)
        riskFree_df = riskFree_df.reindex(dateInterval)
        zhongzheng800quan_df = zhongzheng800quan_df.reindex(dateInterval)

        date = pd.to_datetime(date)
        start = date - pd.to_timedelta(timedelta(weeks=frequency*4))
        if (date in dateInterval) and (start in dateInterval):
            zhongzheng800quan_df = zhongzheng800quan_df[start: date]
            Stock_df = Stock_df[start: date]
            riskFree_df = riskFree_df[start: date]


        regression = pd.DataFrame({
            'x': zhongzheng800quan_df['ROR'],
            'y': Stock_df['S_DQ_CLOSE'] - riskFree_df['AVG_CASH_YTM']
        })

        start = Stock_df.index[0]
        last = Stock_df.index[-1]
        returnOndateInterval = (Stock_df.loc[last, 'S_DQ_CLOSE'] - Stock_df.loc[start, 'S_DQ_CLOSE']) / Stock_df.loc[start, 'S_DQ_CLOSE'] - riskFree_df.loc[last, 'AVG_CASH_YTM']

        res = self.OLS(regression)
        beta = res.params['x']


        return beta, returnOndateInterval

    def crosssectionRegression(self, date):
        CodeSet = self.zhongzheng800StockCodeSet(date)
        regression = pd.DataFrame(columns=['x', 'y'], index=CodeSet)
        for code in regression.index:
            regression.loc[code, 'x'], regression.loc[code, 'y'] = self.timeRegression_1Stock(code, date)

        regression.set_index('code', inplace=True)
        model = self.OLS(regression)
        beta = model.params[1]
        return beta

    def ERP_ts_1factor(self, start, end):
        dateInterval = pd.date_range(start, end)
        ERP_df = pd.DataFrame(None, columns=['ERP'], index = dateInterval)
        for date in ERP_df.index:
            date_str = str(date)
            ERP_df.loc[date, 'ERP'] = self.crosssectionRegression(date_str)

        return ERP_df






    def SizeFactor_ValueFactor(self, date):
        CodeSet = self.zhongzheng800StockCodeSet(date)
        sqlMy = "SELECT `S_INFO_WINDCODE`, `S_VAL_PE_TTM`, `S_VAL_MV`, `S_DQ_CLOSE_TODAY`, `FREE_SHARES_TODAY` FROM ashareeodderivativeindicator WHERE `TRADE_DT` = '" + date + "';"
        dfSource = pd.read_sql(sql=sqlMy, con=self.conn)
        # dfSource.set_index('S_INFO_WINDCODE', inplace=True)
        dfSource['VAL_SHARE'] = dfSource['FREE_SHARES_TODAY'] * 10000 * dfSource['S_DQ_CLOSE_TODAY']

        sqlMyROR = "SELECT `S_INFO_WINDCODE`, `S_DQ_CLOSE`, `S_DQ_PCTCHANGE` FROM ashareeodprices WHERE `TRADE_DT` = '" + date + "';"
        dfROR = pd.read_sql(sql=sqlMyROR, con=self.conn)
        # dfROR.set_index('S_INFO_WINDCODE', inplace=True)


        dfCore = pd.merge(left=dfSource, right=dfROR, how='inner', on='S_INFO_WINDCODE')
        dfCore.set_index('S_INFO_WINDCODE', inplace=True)

        zhongzheng800Stocks_df = dfCore[['S_VAL_PE_TTM', 'S_VAL_MV']].reindex(CodeSet)
        zhongzheng800Factors_df = pd.DataFrame({
            'SizeFactor': 1 / zhongzheng800Stocks_df['S_VAL_PE_TTM'],
            'ValueFactor': zhongzheng800Stocks_df['S_VAL_MV']
        })

        Size_df = zhongzheng800Factors_df['SizeFactor'].sort_values()
        Size_df.fillna(Size_df.mean(), inplace=True)
        SmallSize_df = Size_df[Size_df < Size_df.quantile(.5)]
        BigSize_df = Size_df[Size_df >= Size_df.quantile(.5)]

        Value_df = zhongzheng800Factors_df['ValueFactor'].sort_values()
        Value_df.fillna(Value_df.mean(), inplace=True)
        SmallValue_df = Value_df[Value_df < Value_df.quantile(.3)]
        MiddleValue_df = Value_df[(Value_df >= Value_df.quantile(.3)) & (Value_df < Value_df.quantile(.7))]
        BigValue_df = Value_df[Value_df >= Value_df.quantile(.7)]

        def compute_weightedROR(CodeSetindex):
            resS_S = dfCore[['VAL_SHARE', 'S_DQ_PCTCHANGE']].reindex(CodeSetindex)
            weightedROR = (resS_S['VAL_SHARE'] / resS_S['VAL_SHARE'].sum() * resS_S['S_DQ_PCTCHANGE']).sum()
            return weightedROR


        S_V = compute_weightedROR(SmallSize_df.index & SmallValue_df.index)
        B_V = compute_weightedROR(BigSize_df.index & SmallValue_df.index)
        S_M = compute_weightedROR(SmallSize_df.index & MiddleValue_df.index)
        B_M = compute_weightedROR(BigSize_df.index & MiddleValue_df.index)
        S_G = compute_weightedROR(SmallSize_df.index & BigValue_df.index)
        B_G = compute_weightedROR(BigSize_df.index & BigValue_df.index)


        SizeFactor = 1 / 3 * (S_V + S_M + S_G) - 1 / 3 * (B_V + B_M + B_G)
        ValueFactor = 1 / 2 * (S_V + B_V) - 1 / 2 * (S_G + B_G)

        return SizeFactor, ValueFactor



    def timeRegression_3Factor_1Stock(self, CodeStock, date, frequency=1):
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM aindexeodprices WHERE S_INFO_WINDCODE = 'H00906.CSI';"
        zhongzheng800quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'])
        zhongzheng800quan_df.set_index('TRADE_DT', inplace=True)
        zhongzheng800quan_df.sort_index(inplace=True)

        zhongzheng800quan_df['ROR'] = zhongzheng800quan_df['S_DQ_CLOSE'].pct_change()
        zhongzheng800quan_df.fillna(0, inplace=True)

        riskFree_df, _ = self.bond()

        sqlStock = "SELECT TRADE_DT, S_DQ_CLOSE FROM ashareeodprices WHERE S_INFO_WINDCODE = '" + str(CodeStock) + "';"
        Stock_df = pd.read_sql(sql=sqlStock, con=self.conn, parse_dates=['TRADE_DT'])
        Stock_df.set_index('TRADE_DT', inplace=True)
        Stock_df.sort_index(inplace=True)

        dateInterval = (riskFree_df.index & Stock_df.index) & zhongzheng800quan_df.index

        Stock_df = Stock_df.reindex(dateInterval)
        riskFree_df = riskFree_df.reindex(dateInterval)
        zhongzheng800quan_df = zhongzheng800quan_df.reindex(dateInterval)

        date = pd.to_datetime(date)
        start = date - pd.to_timedelta(timedelta(weeks=frequency * 4))
        if (date in dateInterval) and (start in dateInterval):
            zhongzheng800quan_df = zhongzheng800quan_df[start: date]
            Stock_df = Stock_df[start: date]
            riskFree_df = riskFree_df[start: date]

        regression = pd.DataFrame({
            'x1': zhongzheng800quan_df['ROR'],
            'x2': None,
            'x3': None,
            'y': Stock_df['S_DQ_CLOSE'] - riskFree_df['AVG_CASH_YTM']
        })


        # step 2: y
        start = Stock_df.index[0]
        last = Stock_df.index[-1]
        returnOndateInterval = (Stock_df.loc[last, 'S_DQ_CLOSE'] - Stock_df.loc[start, 'S_DQ_CLOSE']) / Stock_df.loc[start, 'S_DQ_CLOSE'] - riskFree_df.loc[last, 'AVG_CASH_YTM']

        for date in regression.index:
            date_str = timestampTostr(date)
            regression.loc[date, 'x2'], regression.loc[date, 'x3'] = self.SizeFactor_ValueFactor(date_str)

        regression = regression.astype(dtype=float)
        y = np.array(regression['y'].values)

        X = np.array(regression[['x1', 'x2', 'x3']].values)

        model = sm.OLS(y, sm.add_constant(X)).fit()

        # step 2: x1, x2, x3
        beta_zhongzheng800quan = model.params[1]
        beta_sizefactor = model.params[2]
        beta_valuefactor = model.params[3]


        return beta_zhongzheng800quan, beta_sizefactor, beta_valuefactor, returnOndateInterval


# 2.4 时间序列回归与股权风险溢价
class ERP2_4:
    def __init__(self):
        self.conn = DBconnection('wind')

    @staticmethod
    def OLS(x_ts, y_ts):
        x = x_ts.values
        y = y_ts.values
        model = sm.OLS(y, sm.add_constant(x)).fit()
        return model

    def timeseriesOLS(self, x_data, y_data):
        ols_matrix_descirbe = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)

        for y_ts_tmp, column_name in zip(y_data.items(), y_data.columns):
            y_column = []
            for x_ts_tmp in x_data.items():
                y_ts = y_ts_tmp[1]
                x_ts = x_ts_tmp[1]
                y_column.append(self.OLS(x_ts, y_ts))
            ols_matrix_descirbe[column_name] = y_column

        # t-value 矩阵
        # ols_matrix_descirbe_t_slope = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)
        # for column_name in y_data.columns:
        #     for row_name in x_data.columns:
        #         ols_matrix_descirbe_t_slope.loc[row_name, column_name] = \
        #             ols_matrix_descirbe.loc[row_name, column_name].tvalues['x']

        ols_matrix_descirbe_p_slope = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)
        for column_name in y_data.columns:
            for row_name in x_data.columns:
                ols_matrix_descirbe_p_slope.loc[row_name, column_name] = \
                    ols_matrix_descirbe.loc[row_name, column_name].pvalues[1]
        return ols_matrix_descirbe_p_slope

    # 中债国债10年期: CBA04501
    # 中债国债2年期: CBA04601 3-5年期
    # 中债国债3个月: CBA07301 0-3个月
    def bond(self):
            '''
            bond3_5_df: YTM
            bond10_df: YTM
            :return:
            '''

            sqlMy1 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA04501.CS';"
            sqlMy2 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA04601.CS';"
            sqlMy3 = "SELECT TRADE_DT, AVG_CASH_YTM FROM cbondindexeodcnbd WHERE S_INFO_WINDCODE = 'CBA07301.CS';"
            bond10_df = pd.read_sql(sql=sqlMy1, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
            bond3_5_df = pd.read_sql(sql=sqlMy2, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
            bond0_3M_df = pd.read_sql(sql=sqlMy3, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
            bond10_df.sort_index(inplace=True)
            bond3_5_df.sort_index(inplace=True)
            bond0_3M_df.sort_index(inplace=True)

            return bond0_3M_df, bond3_5_df, bond10_df


    def hushen300quan(self):
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM `aindexeodprices` WHERE `S_INFO_WINDCODE` = 'H00300.CSI';"
        hushen300quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        hushen300quan_df.sort_index(inplace=True)
        return hushen300quan_df

    def zhongzheng500quan(self):
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM `aindexeodprices` WHERE `S_INFO_WINDCODE` = 'H00905.CSI';"
        zhongzheng500quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        zhongzheng500quan_df.sort_index(inplace=True)
        return zhongzheng500quan_df

    def zhongzheng800quan(self):
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM `aindexeodprices` WHERE `S_INFO_WINDCODE` = 'H00906.CSI';"
        zhongzheng800quan_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        zhongzheng800quan_df.sort_index(inplace=True)
        return zhongzheng800quan_df

    def chuangyebanIndex(self):
        sqlMy = "SELECT TRADE_DT, S_DQ_CLOSE FROM `aindexeodprices` WHERE `S_INFO_WINDCODE` = '399006.SZ';"
        chuangyebanIndex_df = pd.read_sql(sql=sqlMy, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        chuangyebanIndex_df.sort_index(inplace=True)
        return chuangyebanIndex_df


    # 'H00300.CSI'
    # 'H00905.CSI'
    # 'H00906.CSI'
    # '399006.SZ'

    def dataX_Y_macro(self):
        path = "E:/1-Career/lcmf/Projects/ERP/"
        inflation_df = pd.read_excel(path + "inflation.xls", parse_dates=['date'])
        inflation_df['inflation'] = (inflation_df['CPI:当月同比(月)'] + inflation_df['PPI:全部工业品:当月同比(月)']) / 2
        inflation_df.set_index('date', inplace=True)
        inflation_df.sort_index(inplace=True)
        inflationOnly_df = inflation_df[['inflation']]
        # unit: 1%

        shibor_df = pd.read_excel(path + "Shibor3M.xls", parse_dates=['date'])
        shibor_df.set_index('date', inplace=True)
        shibor_df.sort_index(inplace=True)
        # unit: 1%

        # 中债国债10年期: CBA04501
        # 中债国债2年期: CBA04601 3-5年期
        # 中债国债3个月: CBA07301 0-3个月
        bond0_3M_df, bond3_5_df, bond10_df = self.bond()
        dateInterval_bond = bond0_3M_df.index & (bond3_5_df.index & bond10_df.index)
        timeSpread_df = pd.DataFrame(columns=['期限利差1', '期限利差2'], index=dateInterval_bond)
        timeSpread_df['期限利差1'] = bond10_df.reindex(dateInterval_bond)['AVG_CASH_YTM'] - \
                                 bond3_5_df.reindex(dateInterval_bond)['AVG_CASH_YTM']
        timeSpread_df['期限利差2'] = bond10_df.reindex(dateInterval_bond)['AVG_CASH_YTM'] - \
                                 bond0_3M_df.reindex(dateInterval_bond)['AVG_CASH_YTM']
        # unit: 1%

        #  信用利差 1 和信用利差 2
        creditSpread_df = pd.read_excel(path + "信用利差AAA和AA.xls", parse_dates=['date'])
        creditSpread_df.set_index('date', inplace=True)
        creditSpread_df.sort_index(inplace=True)
        creditSpread_df = creditSpread_df / 100
        creditSpread_df.rename(columns={'信用利差(中位数):产业债AA': '信用利差1', '信用利差(中位数):产业债AAA': '信用利差2'}, inplace=True)
        # unit: 1%

        # inflationOnly_df 是月度数据
        dateInterval_x = (bond10_df.index & timeSpread_df.index) & (shibor_df.index & creditSpread_df.index)
        inflationOnly_df = inflationOnly_df.reindex(dateInterval_x)
        inflationOnly_df.fillna(method='ffill', inplace=True)
        inflationOnly_df.fillna(method='bfill', inplace=True)
        x_df = pd.DataFrame({
            '通胀水平': inflationOnly_df.reindex(dateInterval_x)['inflation'],
            '货币市场利率': shibor_df.reindex(dateInterval_x)['SHIBOR:3个月'],
            '长端利率': bond10_df.reindex(dateInterval_x)['AVG_CASH_YTM'],
            '期限利差1': timeSpread_df.reindex(dateInterval_x)['期限利差1'],
            '期限利差2': timeSpread_df.reindex(dateInterval_x)['期限利差2'],
            '信用利差1': creditSpread_df.reindex(dateInterval_x)['信用利差1'],
            '信用利差2': creditSpread_df.reindex(dateInterval_x)['信用利差2']
        })

        # 沪深300
        hushen300quan_df = self.hushen300quan()
        # 中证500
        zhongzheng500_df = self.zhongzheng500quan()
        # 中证800
        zhongzheng800_df = self.zhongzheng800quan()
        # 创业板指数
        chuangyebanIndex_df = self.chuangyebanIndex()

        dateInterval_y = (hushen300quan_df.index & zhongzheng500_df.index) & (
                    zhongzheng800_df.index & chuangyebanIndex_df.index)

        y_df = pd.DataFrame({
            '沪深300指数': hushen300quan_df.reindex(dateInterval_y)['S_DQ_CLOSE'],
            '中证500指数': zhongzheng500_df.reindex(dateInterval_y)['S_DQ_CLOSE'],
            '中证800指数': zhongzheng800_df.reindex(dateInterval_y)['S_DQ_CLOSE'],
            '创业板指数': chuangyebanIndex_df.reindex(dateInterval_y)['S_DQ_CLOSE']
        })

        y_df = y_df.apply(lambda x: x.pct_change()).fillna(0)

        dateInterval = x_df.index & y_df.index
        x_df = x_df.reindex(dateInterval)
        y_df = y_df.reindex(dateInterval)

        return x_df, y_df


    def data_finance(self, code='H00300.CSI'):
        sql1 = "SELECT TRADE_DT, PE_TTM, DIVIDEND_YIELD, PB_LF FROM aindexvaluation WHERE S_INFO_WINDCODE = '" + code + "';"
        sql2 = "SELECT TRADE_DT, S_DQ_CLOSE FROM aindexeodprices WHERE S_INFO_WINDCODE = '" + code + "';"

        df1 = pd.read_sql(sql1, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        df1.sort_index(inplace=True)
        df2 = pd.read_sql(sql2, con=self.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
        df2.sort_index(inplace=True)

        dateInterval = df1.index & df2.index
        df1 = df1.reindex(dateInterval)
        df2 = df2.reindex(dateInterval)

        df1['每股收益'] = df2['S_DQ_CLOSE'] / df1['PE_TTM']
        df1.rename(columns={'PE_TTM': '市盈率', 'DIVIDEND_YIELD': '股息率', 'PB_LF': '市净率'}, inplace=True)

        return df1

    def financeOLS(self):
        hushen300quan = 'H00300.CSI'
        zhongzheng500quan = 'H00905.CSI'
        zhongzheng800quan = 'H00906.CSI'
        chuangyebanIndex = '399006.SZ'

        hushen300quan_df = self.data_finance(code=hushen300quan)
        zhongzheng500quan_df = self.data_finance(code=zhongzheng500quan)
        zhongzheng800quan_df = self.data_finance(code=zhongzheng800quan)
        chuangyebanIndex_df = self.data_finance(code=chuangyebanIndex)

        _, y_df = self.dataX_Y_macro()

        def vecOLS(x_df, y_ts):
            dateInterval = x_df.index & y_ts.index

            x_df = x_df.reindex(dateInterval)
            y_ts = y_ts.reindex(dateInterval)

            dateInterval = x_df.index & y_ts.index

            x_df = x_df.reindex(dateInterval)
            y_ts = y_ts.reindex(dateInterval)

            vec = pd.Series(index=x_df.columns)

            for index, column in zip(vec.index, x_df.columns):
                model = self.OLS(x_ts=x_df[column], y_ts=y_ts)
                vec[index] = model.pvalues[1]

            return vec

        hushen300quan_vec = vecOLS(hushen300quan_df, y_df['沪深300指数'])
        zhongzheng500quan_vec = vecOLS(zhongzheng500quan_df, y_df['中证500指数'])
        zhongzheng800quan_vec = vecOLS(zhongzheng800quan_df, y_df['中证800指数'])
        chuangyebanIndex_vec = vecOLS(chuangyebanIndex_df, y_df['创业板指数'])

        res_df = pd.DataFrame(columns=y_df.columns, index=hushen300quan_vec.index)
        res_df['沪深300指数'] = hushen300quan_vec
        res_df['中证500指数'] = zhongzheng500quan_vec
        res_df['中证800指数'] = zhongzheng800quan_vec
        res_df['创业板指数'] = chuangyebanIndex_vec

        return res_df


    def volOLS(self):
        _, y_df = self.dataX_Y_macro()

        dateSet = []
        for i in range(10):
            for j in range(1, 13):
                if j < 10:
                    dateSet.append('201' + str(i) + '0' + str(j) + '01')
                else:
                    dateSet.append('201' + str(i) + str(j) + '01')
        dateSet_datetime = list(map(parse, dateSet))

        # 计算月波动率
        def VolMonth(start):
            start = pd.to_datetime(start)
            month = relativedelta(months=1)
            slice = y_df[start: start + month - timedelta(days=1)]
            if len(slice) == 0:
                return pd.Series(index=y_df.columns)
            # ts = res.iloc[:,0]
            res = (slice * slice).sum()
            return res

        # 计算月回报
        def ReturnMonth(start):
            start = pd.to_datetime(start)
            month = relativedelta(months=1)
            slice = y_df[start: start + month - timedelta(days=1)]
            if len(slice) == 0:
                return pd.Series(index=y_df.columns)
            # ts = res.iloc[:,0]
            res = (slice + 1).cumprod()
            return res.iloc[-1, :] - res.iloc[0, :]


        Vol_df = pd.DataFrame(columns=y_df.columns, index=dateSet_datetime)
        Return_df = pd.DataFrame(columns=y_df.columns, index=dateSet_datetime)

        for index, start in zip(Vol_df.index, dateSet):
            Vol_df.loc[index, :] = VolMonth(start)
            Return_df.loc[index, :] = ReturnMonth(start)

        Vol_df.dropna(inplace=True)
        Vol_df = Vol_df.astype(dtype=float)
        Return_df.dropna(inplace=True)
        Return_df = Return_df.astype(dtype=float)

        # 时间区间1: 20100601 - 20141201
        Vol_df_1 = Vol_df[Vol_df.index < '20150101']
        Return_df_1 = Return_df[Return_df.index < '20150101']
        column1 = pd.Series(index=Vol_df.columns)
        for index, column in zip(column1.index, Vol_df_1.columns):
            model = ins.OLS(Vol_df_1[column], Return_df_1[column])
            column1[index] = model.pvalues[1]

        # 时间区间2: 20150101 - 20190701
        Vol_df_2 = Vol_df[Vol_df.index >= '20150101']
        Return_df_2 = Return_df[Return_df.index >= '20150101']
        column2 = pd.Series(index=Vol_df.columns)
        for index, column in zip(column2.index, Vol_df_2.columns):
            model = ins.OLS(Vol_df_2[column], Return_df_2[column])
            column2[index] = model.pvalues[1]

        res_df = pd.DataFrame(columns=['20100601-20141201', '20150101-20190701'], index=Vol_df.columns)
        res_df.iloc[:, 0] = column1
        res_df.iloc[:, 1] = column2

        return res_df



# -------------------------------------------------------------------------------
# Test
ins1 = ERP2_1()
ins2 = ERP2_2()
ins3 = ERP2_3()
ins4 = ERP2_4()








# vol_df = ins.volOLS()




























# _, y_df = ins.dataX_Y_macro()
#
# dateSet = []
# for i in range(10):
#     for j in range(1,13):
#         if j < 10:
#             dateSet.append('201' + str(i) + '0' + str(j) + '01')
#         else:
#             dateSet.append('201' + str(i) + str(j) + '01')
# dateSet_datetime = list(map(parse, dateSet))
#
#
# def VolMonth(start):
#     start = pd.to_datetime(start)
#     month = relativedelta(months=1)
#     slice = y_df[start: start+month-timedelta(days=1)]
#     if len(slice) == 0:
#         return pd.Series(index=y_df.columns)
#     # ts = res.iloc[:,0]
#     res = (slice * slice).sum()
#     return res
#
# def ReturnMonth(start):
#     start = pd.to_datetime(start)
#     month = relativedelta(months=1)
#     slice = y_df[start: start + month - timedelta(days=1)]
#     if len(slice) == 0:
#         return pd.Series(index=y_df.columns)
#     # ts = res.iloc[:,0]
#     res = (slice + 1).cumprod()
#     return res.iloc[-1, :] - res.iloc[0, :]
#
#
# # res = VolMonth('20100101')
#
# Vol_df = pd.DataFrame(columns=y_df.columns, index=dateSet_datetime)
# Return_df = pd.DataFrame(columns=y_df.columns, index=dateSet_datetime)
#
#
# for index,start in zip(Vol_df.index, dateSet):
#     Vol_df.loc[index, :] = VolMonth(start)
#     Return_df.loc[index, :] = ReturnMonth(start)
#
# Vol_df.dropna(inplace=True)
# Vol_df = Vol_df.astype(dtype=float)
# Return_df.dropna(inplace=True)
# Return_df = Return_df.astype(dtype=float)
#
#
# Vol_df_1 = Vol_df[Vol_df.index < '20150101']
# Return_df_1 = Return_df[Return_df.index < '20150101']
# column1 = pd.Series(index=Vol_df.columns)
# for index, column in zip(column1.index, Vol_df_1.columns):
#     model = ins.OLS(Vol_df_1[column], Return_df_1[column])
#     column1[index] = model.pvalues[1]
#
#
# Vol_df_2 = Vol_df[Vol_df.index >= '20150101']
# Return_df_2 = Return_df[Return_df.index >= '20150101']
# column2 = pd.Series(index=Vol_df.columns)
# for index, column in zip(column2.index, Vol_df_2.columns):
#     model = ins.OLS(Vol_df_2[column], Return_df_2[column])
#     column2[index] = model.pvalues[1]
#
# res_df = pd.DataFrame(columns=['20100601-20141201', '20150101-20190701'], index=Vol_df.columns)
# res_df.iloc[:,0] = column1
# res_df.iloc[:,1] = column2












# res_df = ins.financeOLS()
# x_df, y_df = ins.dataX_Y()

# code = 'H00300.CSI'
# sql1 = "SELECT TRADE_DT, PE_TTM, DIVIDEND_YIELD, PB_LF FROM aindexvaluation WHERE S_INFO_WINDCODE = '" + code + "';"
# sql2 = "SELECT TRADE_DT, S_DQ_CLOSE FROM aindexeodprices WHERE S_INFO_WINDCODE = '" + code + "';"
#
# df1 = pd.read_sql(sql1, con=ins.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
# df1.sort_index(inplace=True)
# df2 = pd.read_sql(sql2, con=ins.conn, parse_dates=['TRADE_DT'], index_col=['TRADE_DT'])
# df2.sort_index(inplace=True)
#
# dateInterval = df1.index & df2.index
# df1 = df1.reindex(dateInterval)
# df2 = df2.reindex(dateInterval)
#
# df1['每股收益'] = df2['S_DQ_CLOSE'] / df1['PE_TTM']
# df1.rename(columns={'PE_TTM': '市盈率', 'DIVIDEND_YIELD': '股息率', 'PB_LF': '市净率'}, inplace=True)
#
#
#
#
# x_df = hushen_x_df.copy()
# y_ts = y_df.iloc[:,0]
#
# dateInterval = x_df.index & y_ts.index
#
# x_df = x_df.reindex(dateInterval)
# y_ts = y_ts.reindex(dateInterval)
#
# vec = pd.Series(index=x_df.columns)
#
# for index, column in zip(vec.index, x_df.columns):
#     model = ins.OLS(x_ts=x_df[column], y_ts=y_ts)
#     vec[index] = model.pvalues[1]














# path = "E:/1-Career/lcmf/Projects/ERP/"
# inflation_df = pd.read_excel(path+"inflation.xls", parse_dates=['date'])
# inflation_df['inflation'] = (inflation_df['CPI:当月同比(月)'] + inflation_df['PPI:全部工业品:当月同比(月)'])/2
# inflation_df.set_index('date', inplace=True)
# inflation_df.sort_index(inplace=True)
# inflationOnly_df = inflation_df[['inflation']]
# # unit: 1%
#
# shibor_df = pd.read_excel(path+"Shibor3M.xls", parse_dates=['date'])
# shibor_df.set_index('date', inplace=True)
# shibor_df.sort_index(inplace=True)
# # unit: 1%
#
# # 中债国债10年期: CBA04501
# # 中债国债2年期: CBA04601 3-5年期
# # 中债国债3个月: CBA07301 0-3个月
# bond0_3M_df, bond3_5_df, bond10_df = ins.bond()
# dateInterval_bond = bond0_3M_df.index & (bond3_5_df.index & bond10_df.index)
# timeSpread_df = pd.DataFrame(columns=['期限利差1', '期限利差2'], index=dateInterval_bond)
# timeSpread_df['期限利差1'] = bond10_df.reindex(dateInterval_bond)['AVG_CASH_YTM'] - bond3_5_df.reindex(dateInterval_bond)['AVG_CASH_YTM']
# timeSpread_df['期限利差2'] = bond10_df.reindex(dateInterval_bond)['AVG_CASH_YTM'] - bond0_3M_df.reindex(dateInterval_bond)['AVG_CASH_YTM']
# # unit: 1%
#
# #  信用利差 1 和信用利差 2
# creditSpread_df = pd.read_excel(path+"信用利差AAA和AA.xls", parse_dates=['date'])
# creditSpread_df.set_index('date', inplace=True)
# creditSpread_df.sort_index(inplace=True)
# creditSpread_df = creditSpread_df/100
# creditSpread_df.rename(columns={'信用利差(中位数):产业债AA':'信用利差1', '信用利差(中位数):产业债AAA': '信用利差2'}, inplace=True)
# # unit: 1%
#
# # inflationOnly_df 是月度数据
# dateInterval_x =  (bond10_df.index & timeSpread_df.index) & (shibor_df.index & creditSpread_df.index)
# inflationOnly_df = inflationOnly_df.reindex(dateInterval)
# inflationOnly_df.fillna(method='ffill', inplace=True)
# inflationOnly_df.fillna(method='bfill', inplace=True)
# x_df = pd.DataFrame({
#     '通胀水平': inflationOnly_df.reindex(dateInterval_x)['inflation'],
#     '货币市场利率': shibor_df.reindex(dateInterval_x)['SHIBOR:3个月'],
#     '长端利率': bond10_df.reindex(dateInterval_x)['AVG_CASH_YTM'],
#     '期限利差1': timeSpread_df.reindex(dateInterval_x)['期限利差1'],
#     '期限利差2': timeSpread_df.reindex(dateInterval_x)['期限利差2'],
#     '信用利差1': creditSpread_df.reindex(dateInterval_x)['信用利差1'],
#     '信用利差2': creditSpread_df.reindex(dateInterval_x)['信用利差2']
# })
#
#
#
#
#
# # 沪深300
# hushen300quan_df = ins.hushen300quan()
# # 中证500
# zhongzheng500_df = ins.zhongzheng500quan()
# # 中证800
# zhongzheng800_df = ins.zhongzheng800quan()
# # 创业板指数
# chuangyebanIndex_df = ins.chuangyebanIndex()
#
# dateInterval_y = (hushen300quan_df.index & zhongzheng500_df.index) & (zhongzheng800_df.index & chuangyebanIndex_df.index)
#
# y_df = pd.DataFrame({
#     '沪深300指数': hushen300quan_df.reindex(dateInterval_y)['S_DQ_CLOSE'],
#     '中证500指数': zhongzheng500_df.reindex(dateInterval_y)['S_DQ_CLOSE'],
#     '中证800指数': zhongzheng800_df.reindex(dateInterval_y)['S_DQ_CLOSE'],
#     '创业板指数': chuangyebanIndex_df.reindex(dateInterval_y)['S_DQ_CLOSE']
# })
#
#
# y_df = y_df.apply(lambda x: x.pct_change()).fillna(0)
#
#
# dateInterval = x_df.index & y_df.index
# x_df = x_df.reindex(dateInterval)
# y_df = y_df.reindex(dateInterval)




















# ins = ERP2_3()
#
#
#
# date = '20190701'
# CodeStock = '000415.SZ'
# frequency = 1
#
# # res = ins.timeRegression_3Factor_1Stock(date=date, CodeStock=CodeStock)
#
# CodeSet = ins.zhongzheng800StockCodeSet(date)
# regression = pd.DataFrame(columns=['x1', 'x2', 'x3', 'y'], index=CodeSet)
# for code in regression.index:
#     regression.loc[code, 'x1'], regression.loc[code, 'x2'], regression.loc[code, 'x3'], regression.loc[code, 'y'] = ins.timeRegression_3Factor_1Stock(code, date)
#
# regression.astype(dtype=float)
# # regression.fillna(method='ffill', inplace=True)
# y = np.array(regression['y'].values)
# X = np.array(regression[['x1', 'x2', 'x3']].values)
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# beta = model.params[1]




# regression.set_index('code', inplace=True)
#
#
# model = ins.OLS(regression)
# beta = model.params[1]
# return beta

# ExcelWriter = pd.ExcelWriter(path="E:/1-Career/lcmf/Projects/ERP/regression_3factor.xls")
# regression.to_excel(ExcelWriter, sheet_name='cross_regression_3factor')
# ExcelWriter.save()
#
# ExcelWriter.close()





# path = "E:/1-Career/lcmf/Projects/ERP"
# dfCore = pd.read_excel(path+"/inflation.xls", parse_dates=['date'])
# dfCore.set_index('date', inplace=True)
















# df1, df2 = ins.align_nominal_2111()
# fig = plt.figure(figsize=(8,6))
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()


# method 1
# data1 = df1[['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM']]
# data1.plot(ax=ax1)
# df1['hushen300Future1YReturn'].plot(ax=ax2, kind='line')
#
# plt.show()


# method 2
# l1 = ax1.plot(df1['hushen300DividendRate-bond3_5YTM'].index, df1['hushen300DividendRate-bond3_5YTM'].values, 'r-', label='hushen300DividendRate-bond3_5YTM')
# l2 = ax1.plot(df1['hushen300DividendRate-bond10YTM'].index, df1['hushen300DividendRate-bond10YTM'].values, 'b-', label='hushen300DividendRate-bond10YTM')
# l3 = ax2.plot(df1['hushen300Future1YReturn'].index, df1['hushen300Future1YReturn'].values, 'g-', label='hushen300Future1YReturn')
#
# plt.legend(handles=[l1, l2, l3], labels=['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM', 'hushen300Future1YReturn'])
# ax1.set_ylabel('%')
# ax2.set_ylabel('%')
# ax1.set_xlabel('年份')
# ax1.set_title('title')
# plt.show()

# method 3
# df1.astype(dtype=float)
# df1[['hushen300DividendRate-bond3_5YTM', 'hushen300DividendRate-bond10YTM']].plot()
# df1['hushen300Future1YReturn'].plot(secondary_y=True)
# ax = df1.plot(secondary_y=['hushen300Future1YReturn'])
# plt.show()



# f1 = ins.DDM4_2132()
# ins.figure1Plot_2121()
# ins.figure12Plot_2122()
# ins.figure1Plot_2132()



#
# ins = ERP2_2()
# df = ins.history()
# ins.figure1Plot_22()
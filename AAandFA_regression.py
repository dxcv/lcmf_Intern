import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm

from datetime import datetime
from datetime import timedelta

def weekFilter(df, start=0):
    df_adjust = df.iloc[start:]
    return df_adjust.iloc[::5]

# .rolling 也是另一种处理窗口数据的方法
def backRoll(df, day, weekInterval=26):
    if day in df.index[:weekInterval]:
        return None
    else:
        for i in range(len(df.index)):
            if df.index[i] == day:
                break
        return df.iloc[i-weekInterval:i]


data_path = 'E:/1-Career/lcmf/factorModel'

tradeDays_WeekEnd = pd.read_excel(data_path+'/mofang_TRADE_DAYS.xls', sheet_name='周收盘日')
tradeDays_WeekEnd.set_index('td_date', inplace=True)
x_df = pd.read_excel(data_path+'/CoreData.xls', sheet_name='x_df')
x_df.set_index('TRADE_DAYS', inplace=True)
y_df = pd.read_excel(data_path+'/CoreData.xls', sheet_name='y_return_df')
y_df.set_index('TRADE_DAYS', inplace=True)
# y_df 从周一开始

# 取每周周五的数据
x_week_df = x_df.reindex(tradeDays_WeekEnd.index & x_df.index)
y_week_df = y_df.reindex(tradeDays_WeekEnd.index & y_df.index)

# tmp = backRoll(y_week_df, datetime(2019,7,12))
# print(len(tmp))




# demo
# tmp = pd.DataFrame({'x':x_df.iloc[:,0],
#                     'y':y_df.iloc[:,0]})
# model = ols('y~x', tmp).fit()
# print(model.summary())
#
#
# print(sm.tsa.stattools.adfuller(x_df.iloc[:,0]))
# print(type(sm.tsa.stattools.adfuller(x_df.iloc[:,0])))



# ADF检验时间序列的平稳性
def tupleToDataFrame(adf_tuple, key):
    dict1 = {
                'T统计量': adf_tuple[0],
                'P值': adf_tuple[1],
            }
    dict2 = dict(dict1, **adf_tuple[4])
    return pd.DataFrame(dict2, index=[key])

def ADFtest_list(list_ts, list_names):
    adf_tuple_tmp = sm.tsa.stattools.adfuller(list_ts[0])
    df = tupleToDataFrame(adf_tuple_tmp, list_names[0])

    if len(list_names) > 1:
        for ts, name in zip(list_ts[1:], list_names[1:]):
            adf_tuple_tmp = sm.tsa.stattools.adfuller(ts)
            df = df.append(tupleToDataFrame(adf_tuple_tmp, name))
    return df

ADF_df = ADFtest_list([ts[1] for ts in x_df.items()], [name for name in x_df])



def StressTest():
    count = 1
    while 1:
        tmp = backRoll(y_week_df, datetime(2019, 7, 12), count)
        print(count, len(tmp))
        if count != len(tmp) or count > 300:
            return

        count += 1


def OLS(x_input, y_input):
    tmp_data = pd.DataFrame({'x':x_input, 'y':y_input})
    model = ols('y~x', tmp_data).fit()
    return model





# ols_matrix_descirbe = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)
#
# for y_ts_tmp, column_name in zip(y_data.items(), y_data.columns):
#     y_column = []
#     for x_ts_tmp in x_data.items():
#         y_ts = y_ts_tmp[1]
#         x_ts = x_ts_tmp[1]
#         y_column.append(OLS(x_ts, y_ts))
#     ols_matrix_descirbe[column_name] = y_column
#
#
# ols_matrix_descirbe_t_slope = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)
# for column_name in y_data.columns:
#     for row_name in x_data.columns:
#         ols_matrix_descirbe_t_slope.loc[row_name, column_name] = ols_matrix_descirbe.loc[row_name, column_name].tvalues['x']




# 实验日期： 2019年7月13日
def crossSectionOLS(x_data, y_data):
    ols_matrix_descirbe = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)

    for y_ts_tmp, column_name in zip(y_data.items(), y_data.columns):
        y_column = []
        for x_ts_tmp in x_data.items():
            y_ts = y_ts_tmp[1]
            x_ts = x_ts_tmp[1]
            y_column.append(OLS(x_ts, y_ts))
        ols_matrix_descirbe[column_name] = y_column

    ols_matrix_descirbe_t_slope = pd.DataFrame(None, index=x_data.columns, columns=y_data.columns)
    for column_name in y_data.columns:
        for row_name in x_data.columns:
            ols_matrix_descirbe_t_slope.loc[row_name, column_name] = \
                ols_matrix_descirbe.loc[row_name, column_name].tvalues['x']

    return ols_matrix_descirbe_t_slope


# # part 1: y -- x
# experimentalDate = datetime(2019, 7, 13)
# y_data1 = backRoll(y_week_df, experimentalDate, weekInterval=26)
# x_data1 = backRoll(x_week_df, experimentalDate, weekInterval=26)
#
# # part 2: \Delta y -- x
# experimentalDate = datetime(2019, 7, 13)
# y_data2 = backRoll(y_week_df.rolling(window=2).apply(lambda x: x[1] - x[0], raw=False), experimentalDate, weekInterval=26)
# x_data2 = backRoll(x_week_df, experimentalDate, weekInterval=26)
#
# # part 3: y -- \Delta x
# experimentalDate = datetime(2019, 7, 13)
# y_data3 = backRoll(y_week_df, experimentalDate, weekInterval=26)
# x_data3 = backRoll(x_week_df.rolling(window=2).apply(lambda x: x[1] - x[0], raw=False), experimentalDate, weekInterval=26)
#
# # part 4: \Delta y -- \Delta x
# experimentalDate = datetime(2019, 7, 13)
# y_data4 = backRoll(y_week_df.rolling(window=2).apply(lambda x: x[1] - x[0], raw=False), experimentalDate, weekInterval=26)
# x_data4 = backRoll(x_week_df.rolling(window=2).apply(lambda x: x[1] - x[0], raw=False), experimentalDate, weekInterval=26)
#
# exp_matrix1 = crossSectionOLS(x_data1, y_data1)
# print(exp_matrix1)
#
# exp_matrix2 = crossSectionOLS(x_data2, y_data2)
#
# exp_matrix4 = crossSectionOLS(x_data4, y_data4)
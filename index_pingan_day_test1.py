import pandas as pd
import numpy as np
import pymysql as sql

# Data Processing
#---------------------------
# select index of '601318.SH' from database
conn = sql.Connect(host='192.168.88.12', user='public', passwd='h76zyeTfVqAehr5J', database='wind', charset='utf8')
pingan_raw_sql = "SELECT * FROM ashareintensitytrend WHERE S_INFO_WINDCODE = '601318.SH'"
pingan_raw = pd.read_sql(sql=pingan_raw_sql, con=conn, parse_dates=['TRADE_DT'])

pingan_raw.pop('OBJECT_ID')
pingan_raw.pop('S_INFO_WINDCODE')
pingan_raw.pop('DDI')
pingan_raw.pop('DDI_ADDI')
pingan_raw.pop('DDI_AD')
pingan_raw.pop('OPDATE')
pingan_raw.pop('OPMODE')

# should be 30 columns

pingan_raw.set_index('TRADE_DT', inplace=True)
# type of date is 'DatetimeIndex'

pingan_raw.sort_index(inplace=True)
# begin: 2007-02-12    end: 2019-06-07



#---------------------------------------------
# select a stock: 中国平安 code: 601318.SH
excelFile_stock = 'E:\\1-FuturePlan\\Career\\lcmf\\python\\中国平安.xlsx'
Stock_raw = pd.read_excel(excelFile_stock, index_col=0, parse_dates=['Date'])
# type of date is 'DatetimeIndex'

Stock_raw.pop('open')
Stock_raw.pop('high')
Stock_raw.pop('low')
Stock_raw.pop('volume')
Stock_raw.dropna(inplace=True)
Stock_raw.sort_index()

Stock_pct = Stock_raw.copy()
Stock_pct['close'] = Stock_pct['close'].pct_change()
Stock_pct = Stock_pct.drop(Stock_pct.index[0])

#------------------------------
# generate the label dataframe
def binary_filter(x):
    if x.iloc[0] > 0:
        return 1
    else:
        return 0
#-----------------------------
Stock_label = Stock_pct.apply(binary_filter, axis=1)
# begin: 2007-03-02    end: 2019-06-06





#-------------------------------------------------
# alignment
# pingan_raw is longer than Stock_label
pingan_index = pingan_raw.copy()
pingan_index = pingan_index.reindex(Stock_label.index)






#------------------------------------------
# input:  pingan_index, Stock_pct
def null_detector(df):
    res = pd.DataFrame(None, index=df.columns)
    res['IsNull'] = df.isnull().any()
    res['NumOfNull'] = df.isnull().sum()
    res['Percentage'] = df.isnull().sum()/len(df)

    return res

# null_stat_df = null_detector(pingan_raw)
# max null percent is 5%

df1 = null_detector(pingan_index)
df2 = null_detector(Stock_pct)



#----------------------------------------
# Processing missig data
# truncate the first 142 rows
pingan_index = pingan_index.iloc[143:,]
Stock_label = Stock_label.iloc[143:,]

# still missing data in the middle of 'pingan_index'
pingan_index = pingan_index.fillna(method='bfill')


# data: 2483


#--------------
# define random generator
# left or right stdev
# Structure of df(parameter):
def random_generator(df):
    df_copy = df.copy()
    for col, series in df.iteritems():
        std = series.std()
        variation_ser_ = pd.Series((2*np.random.rand(len(df))-1)*std, index=df.index)
        df_copy[col] = df_copy[col] + variation_ser_

    return df_copy




# Input: pingan_index, Stock_label
# xgboost module
import xgboost as xgb

#--------------------------------
# 1. split the date into pieces with ratio 5:2:3
ratio_mark_temp = [round(len(Stock_label) * 0.5), round(len(Stock_label) * 0.2), round(len(Stock_label) * 0.3)]
ratio_mark = [ratio_mark_temp[0], ratio_mark_temp[0] + ratio_mark_temp[1],
                  ratio_mark_temp[0] + ratio_mark_temp[1] + ratio_mark_temp[2]]
# Store the real value of test set
real_test_label = Stock_label[ratio_mark[1]:ratio_mark[2]]
#---------------------------------


#---------------------------------
# 2. xgboost train-predict function
def xgb_pred(df_data, df_label):

    # transform the data into DMatrix type
    dtrain = xgb.DMatrix(df_data[:ratio_mark[0]], label=df_label[:ratio_mark[0]])

    dvalid = xgb.DMatrix(df_data[ratio_mark[0]:ratio_mark[1]], label=df_label
    [ratio_mark[0]:ratio_mark[1]])

    dtest = xgb.DMatrix(df_data[ratio_mark[1]:ratio_mark[2]], label=df_label
    [ratio_mark[1]:ratio_mark[2]])

    # Explanation for parameters:
    # max_depth: maximum depth; increasing this value will make the model more complex and more likely to overfit, range: [0,∞]
    # eta: like learning rate, range: [0,1]
    # gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree, range: [0,∞]
    # silent: whether output the intermediate result
    # objective: ensure type of objective function
    # nthread: maximum thread to be used
    # eval_metric:
    # use space+\ to mullti-coding

    # ------------------------------
    # Parameter 1:
    param = {
        'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 4,
        'eval_metric': 'auc'
    }

    # # ------------------------------
    # # Parameter 2:
    # param = {
    #     'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 4,
    #     'eval_metric': 'auc'
    # }

    # # ------------------------------
    # # Parameter 3:
    # param = {
    #     'max_depth': 3, 'eta': 0.5, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 4,
    #     'eval_metric': 'auc'
    # }

    # Parameter choice explanation
    # ---------------------------------
    # Empirical results of Para. 1-3
    #
    # Similarity: train-auc always equals 1 except the first rounds, whose value is still high enough over 0.90
    #
    # Difference: eval-auc
    #             para_1 > para_2 > para_3
    #
    #
    # ---------------------------------







    # num_round: total times of training
    num_round = 10000

    evallist = [(dvalid, 'eval'), (dtrain, 'train')]

    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=200)

    # Test using data from training set
    # Given data, the prediction is stable.
    ypred = bst.predict(dtest)
    return ypred

#---------------------------------
# 3. Execution
def main_loop(macro, index_label, total_rounds):
    # Preparation befor main loop
    # using 'xgb_pred' to generate dataframe with row be round of training and column be order number of test set
    pred_df = pd.DataFrame(columns=[i for i in range(ratio_mark_temp[-1]-1)])

    for i in range(total_rounds):
        ds_macro_day = random_generator(macro)
        ser_ds = pd.Series(xgb_pred(ds_macro_day, index_label))
        pred_df = pred_df.append(ser_ds, ignore_index=True)

    return pred_df

#-------------------------------------------------------------------------
# 4. Measure of stability
# JS_div(ds1, ds2, num_mesh)
# 1). Kullback-Leibler divergence: not commutative
def KL_div(ds1, ds2, num_mesh):
    prob1 = discrete_prob(ds1, num_mesh)
    prob2 = discrete_prob(ds2, num_mesh)

    # as long as 'num_mesh' are identical, the asymptotic discrete pdfs lie in the same set of intervals
    # using the formula
    res = 0
    for i in range(len(prob1)):
        if prob1[i] == 0:
            continue
        elif prob2[i] == 0:
            res += prob1[i] * (np.log(prob1[i]) - np.log(0.001))
        else:
            res += prob1[i] * (np.log(prob1[i]) - np.log(prob2[i]))

    return res


# 2). Jensen-Shannon divergence: commutative
def JS_div(ds1, ds2, num_mesh):
    ds_avg = (ds1 + ds2)/2
    res = 0.5*KL_div(ds1, ds_avg, num_mesh) + 0.5*KL_div(ds2, ds_avg, num_mesh)
    return res


def discrete_prob(ds, num_mesh):
    # ds.values lies in [0,1]
    num_round = len(ds)
    L = list(np.linspace(0,1,num_mesh))
    pdf = [];
    for i in range(len(L)-1):
        mask = (ds > L[i]) & (ds < L[i+1])
        pdf.append(len(ds[mask])/num_round)

    # num of interval is 'num_mesh'-1
    # pdf is a list
    return pdf

def JS_mat(L, num_mesh):
    # L is a list of ds
    res = np.zeros((len(L),len(L)))
    for i in range(len(L)):
        for j in range(i,len(L)):
            temp = JS_div(L[i], L[j], num_mesh)
            res[i][j] = temp

    return res




#--------------------------------------------------------------------------------

# main function
num_round = 5
num_pdf = 10
List_df = []
# training
for i in range(num_pdf):
    List_df.append(main_loop(pingan_index, Stock_label, num_round))


def stability(L, index, num_mesh):
    L_ds = []
    for item in L:
        L_ds.append(item.iloc[:,index])

    M = JS_mat(L_ds, num_mesh)
    M_max = M.max()

    return M_max




len_stab_index = len(List_df[0].columns)
stab = []
num_mesh1 = 5
# compute the norm of every upper-triangular matrix
for i in range(5):
    stab.append(stability(List_df, i, num_mesh1))





















import pymysql
import pandas as pd
import numpy as np


# Data Preprocessing
excelFile = u'美国利率.xls'
macro_raw = pd.read_excel(excelFile, index_col=0)
#------------------
# Step0: correct the format of date
# the first index is '频率'
macro_raw = macro_raw.drop(macro_raw.index[0])
macro_raw.index = pd.to_datetime(macro_raw.index)
macro_raw = macro_raw.sort_index()
#---------------
# Step1: erase the week/season data
# week column '美国:所有联储银行:资产:持有证券:抵押贷款支持债券(MBS)'
# season column '美国:GDP:不变价:环比折年率:季调'
macro_raw.drop('美国:所有联储银行:资产:持有证券:抵押贷款支持债券(MBS)', axis=1, inplace=True)
macro_raw.drop('美国:GDP:不变价:环比折年率:季调', axis=1, inplace=True)
#--------------
# process the missing data
macro_raw_inter = macro_raw.copy()
macro_raw_inter = macro_raw_inter.fillna(method='bfill')
macro_raw_inter = macro_raw_inter.fillna(method='ffill')


#--------------
# Step3: define random generator
# left or right stdev
# Structure of df(parameter):
def random_generator(df):
    df_copy = df.copy()
    for col, series in df.iteritems():
        std = series.std()
        variation_ser_ = pd.Series((2*np.random.rand(len(df))-1)*std, index=df.index)
        df_copy[col] = df_copy[col] + variation_ser_

    return df_copy






#---------------------------------------------
# select 标普500 globalid = 120000013
SP500_raw = pd.read_csv('SP500_nav.csv', index_col="ra_date")
SP500_raw = SP500_raw.sort_index()

# use 'pct_change' to compute the percentage change
SP500_inc_trunc = SP500_raw.copy()
SP500_inc_trunc['ra_nav'] = SP500_inc_trunc['ra_nav'].pct_change()
SP500_inc_trunc = SP500_inc_trunc.drop(SP500_inc_trunc.index[0])

#------------------------------------------
# generate the label dataframe
def binary_filter(x):
    if x.iloc[0] > 0:
        return 1
    else:
        return 0
#-----------------------------------------
SP500_label = SP500_inc_trunc.apply(binary_filter, axis=1)
SP500_label.index = pd.to_datetime(SP500_label.index)

# Special process
# you should recheck the index between 'macro_day' and 'SP500_label'
macro_day_trunc = macro_raw_inter.copy()
# macro_day_trunc = macro_day_trunc.to_dat
macro_day_trunc = macro_day_trunc.reindex(SP500_label.index)
macro_day_trunc = macro_day_trunc.fillna(method='bfill')
















# Input: macro_day_trunc, SP500_label
# xgboost module
import xgboost as xgb

#--------------------------------
# 1. split the date into pieces with ratio 5:2:3
ratio_mark_temp = [round(len(SP500_label) * 0.5), round(len(SP500_label) * 0.2), round(len(SP500_label) * 0.3)]
ratio_mark = [ratio_mark_temp[0], ratio_mark_temp[0] + ratio_mark_temp[1],
                  ratio_mark_temp[0] + ratio_mark_temp[1] + ratio_mark_temp[2]]
# Store the real value of test set
real_test_label = SP500_label[ratio_mark[1]:ratio_mark[2]]
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


num_round = 5
num_pdf = 10
List_df = []
# training
for i in range(num_pdf):
    List_df.append(main_loop(macro_day_trunc, SP500_label, num_round))


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






















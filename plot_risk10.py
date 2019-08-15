import pandas as pd
import matplotlib.pyplot as plt

df_risk = pd.read_csv('基准和竞品的数据.csv', index_col=0, encoding='gbk')
# df_risk = pd.read_csv('基准和竞品的数据.csv', index_col=0)
# the first 8 columns are what we are concerned
df_risk = df_risk.iloc[:, :8]

df_user = pd.read_csv('每天购买赎回比例.csv', index_col=0, encoding='gbk')
# the last 4 columns are what we are concerned
df_user = df_user.iloc[:, -4:]


# construct increment percent DataFrame
df_inc = df_risk.copy()
# index_list = list(df_inc.columns)
for column in df_inc.columns:
    df_inc[column] = df_inc[column].pct_change()
# the first row is NaN
df_inc = df_inc.drop(df_inc.index[0])


# truncate the date according to user period
df_inc_user = df_inc[df_user.index[0]:]


def subtraction(x):
    # if positive, it means our product is better than market; otherwise, it means worse.
    return x.iloc[0] - x.iloc[1]


risk3 = df_inc_user[['风险3', '风险3比较基准']]
risk5 = df_inc_user[['风险5', '风险5比较基准']]
risk7 = df_inc_user[['风险7', '风险7比较基准']]
risk10 = df_inc_user[['风险10', '风险10比较基准']]

risk3_mask = risk3.apply(subtraction, axis=1)
risk5_mask = risk5.apply(subtraction, axis=1)
risk7_mask = risk7.apply(subtraction, axis=1)
risk10_mask = risk10.apply(subtraction, axis=1)

# zhongfont = matplotlib.font_manager.FontProperties(fname='C:\\Windows\\Fonts\\STKAITI.TTF')

#---------------------------------------------------------

#risk 10
fig = plt.figure(figsize=(20,20))
ax10_com = fig.add_subplot(211)
ax10_user = fig.add_subplot(212)

risk10_mask = risk10_mask.reindex(df_user.index)
risk10_mask.plot(ax=ax10_com)
ax10_com.set_title('风险10的相对优势')
# ax10_com.legend(['Difference: product - market'], loc='best')
ax10_com.legend(['竞品与市场比较'], loc='best')
ax10_com.set_xlabel('时间')

# df_user.columns = ['Buyer ratio', 'Purchase ratio', 'Seller ratio', 'Retreat ratio']
df_user.plot(ax=ax10_user)
ax10_user.set_title('用户行为')
ax10_user.legend(loc='best')
ax10_user.set_xlabel('时间')

plt.rcParams['font.sans-serif']=['SimHei']
plt.savefig('risk10.png', dpi=400, bbox_inches='tight')
plt.show()



# risk3
fig = plt.figure(figsize=(20,20))
ax10_com = fig.add_subplot(211)
ax10_user = fig.add_subplot(212)

risk3_mask = risk3_mask.reindex(df_user.index)
risk3_mask.plot(ax=ax10_com)
ax10_com.set_title('风险3的相对优势')
# ax10_com.legend(['Difference: product - market'], loc='best')
ax10_com.legend(['竞品与市场比较'], loc='best')
ax10_com.set_xlabel('时间')

# df_user.columns = ['Buyer ratio', 'Purchase ratio', 'Seller ratio', 'Retreat ratio']
df_user.plot(ax=ax10_user)
ax10_user.set_title('用户行为')
ax10_user.legend(loc='best')
ax10_user.set_xlabel('时间')

plt.rcParams['font.sans-serif']=['SimHei']
plt.savefig('risk3.png', dpi=400, bbox_inches='tight')
plt.show()


# risk5
fig = plt.figure(figsize=(20,20))
ax10_com = fig.add_subplot(211)
ax10_user = fig.add_subplot(212)

risk5_mask = risk5_mask.reindex(df_user.index)
risk5_mask.plot(ax=ax10_com)
ax10_com.set_title('风险5的相对优势')
# ax10_com.legend(['Difference: product - market'], loc='best')
ax10_com.legend(['竞品与市场比较'], loc='best')
ax10_com.set_xlabel('时间')

# df_user.columns = ['Buyer ratio', 'Purchase ratio', 'Seller ratio', 'Retreat ratio']
df_user.plot(ax=ax10_user)
ax10_user.set_title('用户行为')
ax10_user.legend(loc='best')
ax10_user.set_xlabel('时间')

plt.rcParams['font.sans-serif']=['SimHei']
plt.savefig('risk5.png', dpi=400, bbox_inches='tight')
plt.show()

# risk7
fig = plt.figure(figsize=(20,20))
ax10_com = fig.add_subplot(211)
ax10_user = fig.add_subplot(212)

risk7_mask = risk7_mask.reindex(df_user.index)
risk7_mask.plot(ax=ax10_com)
ax10_com.set_title('风险7的相对优势')
# ax10_com.legend(['Difference: product - market'], loc='best')
ax10_com.legend(['竞品与市场比较'], loc='best')
ax10_com.set_xlabel('时间')

# df_user.columns = ['Buyer ratio', 'Purchase ratio', 'Seller ratio', 'Retreat ratio']
df_user.plot(ax=ax10_user)
ax10_user.set_title('用户行为')
ax10_user.legend(loc='best')
ax10_user.set_xlabel('时间')

plt.rcParams['font.sans-serif']=['SimHei']
plt.savefig('risk7.png', dpi=400, bbox_inches='tight')
plt.show()


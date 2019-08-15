import pymysql
import pandas as pd

'''
Data Description:       
'''
sql_dict = {
                # 'mz_markowitz': "SELECT * FROM mz_markowitz WHERE globalid = 'MZ.000060';",
                # 'mz_markowitz_alloc': "SELECT * FROM mz_markowitz_alloc WHERE globalid = 'MZ.000060';",
                # 'mz_markowitz_argv': "SELECT * FROM mz_markowitz_argv WHERE mz_markowitz_id = 'MZ.000060';",
                # 'mz_markowitz_asset': "SELECT * FROM mz_markowitz_asset WHERE mz_markowitz_id = 'MZ.000060';",
                # 'mz_highlow': "SELECT * FROM mz_highlow WHERE globalid = 'HL.000060';",
                # 'mz_highlow_alloc': "SELECT * FROM mz_highlow_alloc WHERE globalid = 'HL.000060';",
                # # 'mz_highlow_argv': "SELECT * FROM mz_highlow_argv WHERE globalid = 'HL.000060';",
                # 'mz_highlow_asset': "SELECT * FROM mz_highlow_asset WHERE mz_highlow_id = 'HL.000060';",
                'rm_riskmgr_index_best_start_end': "SELECT * FROM rm_riskmgr_index_best_start_end;",
                'mz_markowitz_bounds': "SELECT * FROM mz_markowitz_bounds;"
            }

conn = pymysql.Connect(host='192.168.88.254', user='root', passwd='Mofang123', database='asset_allocation', charset='utf8')

list_df = []
path = "E:/1-FuturePlan/Career/lcmf/DataMunging_20190718.xls"
ExcelWriter = pd.ExcelWriter(path)



for key, value in sql_dict.items():
    temp = pd.read_sql(sql=value, con=conn)
    temp.pop(temp.columns[-1])
    temp.pop(temp.columns[-1])

    list_df.append(temp)

for (df,key) in zip(list_df, sql_dict.keys()):
    df.to_excel(ExcelWriter, sheet_name=key)

ExcelWriter.save()
ExcelWriter.close()

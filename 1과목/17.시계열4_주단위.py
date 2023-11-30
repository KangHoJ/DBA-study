import pandas as pd 
df= pd.read_csv('./basic2.csv',parse_dates=['Date'],index_col=['Date']) # 인덱스를 Date로 만들고 
print(df.resample('W').sum()) # 주단위
print(df.resample('2W').sum()) # 2주단위
print(df.resample('M').sum()) # 월단위
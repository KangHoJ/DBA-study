import pandas as pd
import numpy as np
df=pd.read_csv("mtcars.csv")
df.head()
#	문제1 o
#	mpg	변수의 제 1사분위수를 구하고 반올림하여 정수값으로 출력하시오
mpg_1 = df['mpg'].quantile(0.25)
print('문제1:',round(mpg_1)) 

#	문제2 o
#	mpg	값이 19이상	21이하인 데이터의 수를 구하시오.
print('문제2:',len(df[(df['mpg']>=19) & (df['mpg']<=21)])) 


#	문제3 o
#	hp	변수의 IQR	값을 구하시오
Q3 = df['hp'].quantile(0.75)
Q1 = df['hp'].quantile(0.25)
IQR = Q3-Q1
print('문제3:',IQR) 

#	문제4 o
#	wt	변수의 상위	10개 값의 총합을 구하여 소수점을 버리고 정수로 출력하시오
df = df['wt'].sort_values(ascending=False).head(10)
print('문제4:',int(sum(df))) 

#	문제5 o
#	전체 자동차에서	cyl가 6인 비율이 얼마인지 반올림하여 소수점 첫째자리까지 출력하시오.
df=pd.read_csv("mtcars.csv")
pr = len(df[df['cyl']==6]) / len(df)
print('문제5:',round(pr,1))

#	문제6 o
#	첫번째 행부터 순서대로 10개 뽑은 후 mpg	열의 평균값을 반올림하여 정수로 출력하시오
df_10 = df.head(10)
print('문제6:',round(df_10['mpg'].mean()))

#	문제7 o 
#	첫번째 행부터 순서대로 50%까지 데이터를 뽑아 wt	변수의 중앙값을 구하시오.
df=pd.read_csv("mtcars.csv")
print('문제7:',df.head(16)['wt'].median())
# 방법 2 
p50	= int(len(df)*0.5)
df50=df[:p50]



#	문제8 o
#	결측값이 있는 데이터의 수를 출력하시오.	
df = pd.read_csv('test.csv')
print('문제8',df.isnull().sum().sum())

#	문제9 o
#	'판매수' 컬럼의 결측값을 판매수의 중앙값으로 대체하고
#	판매수의 평균값을 반올림하여 정수로 출력하시오.	
df = pd.read_csv('test.csv')
df['판매수'] = df['판매수'].fillna(df['판매수'].median())
# print(df['판매수'].isnull().sum())
print('문제9',round(df['판매수'].mean()))

#	문제 10 o
#	판매수 컬럼에 결측치가 있는 행을 제거하고,
#	첫번째 행부터 순서대로	50%까지의 데이터를 추출하여
#	판매수 변수의	Q1(제1사분위수)값을 반올림하여 정수로 출력하시오.	
#	데이터	(수정금지)
df = pd.read_csv('test.csv')
df = df[~df['판매수'].isnull()]
print('문제10',round(df.head(5)['판매수'].quantile(0.25)))



##########################################################################################

df = pd.read_csv("mtcars.csv")
#	문제 11 o
#	cyl가 4인 자동차와 6인 자동차 그룹의mpg	평균값 차이를 절대값으로 반올림하여 정수로 출력하시오
cyl_4 = df[df['cyl']==4]['mpg'].mean()
cyl_6 = df[df['cyl']==6]['mpg'].mean()
print('문제11',round(abs(cyl_6-cyl_4)))

#	문제12 o 
#	hp	변수에 대해 데이터표준화(Z-score)를 진행하고 이상치의 수를 구하시오.
#	(단, 이상치는 Z값이 1.5를 초과하거나 -1.5 미만인 값이다)
sig = np.sqrt(df['hp'].var())
# print(df['hp'])
df['z'] = (df['hp']-df['hp'].mean())/df['hp'].std()
# print(df['z'])
print('문제12',len(df[(df['z']>1.5)|(df['z']<-1.5)]))

#	문제13 o
#	mpg	컬럼을 최소최대	Scaling을 진행한 후	0.7보다 큰 값을 가지는 레코드 수를 구하라
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df['mpg'] = sc.fit_transform(df[['mpg']])
print('문제13',len(df[df['mpg']>0.7]))


#	문제14 o
#	wt컬럼에 대해 상자그림 기준으로 이상치의 개수를 구하시오.	
df = pd.read_csv("mtcars.csv")
Q1 = df['wt'].quantile(0.25)
Q3 = df['wt'].quantile(0.75)
IQR = Q3-Q1
outline1 = df['wt']<Q1-1.5*IQR
outline2 = df['wt']>Q3+1.5*IQR
print('문제14',len(df[outline1 | outline2]))


#	문제15 o
#	판매수 컬럼의 결측치를 최소값으로 대체하고
#	결측치가 있을 때와 최소값으로 대체했을 때 평균값의 차이를 절대값으로 반올림하여 정수로 출력하시오
#	데이터 생성	(수정금지)
df = pd.read_csv('test.csv')
before = df['판매수'].mean()
df['판매수'] = df['판매수'].fillna(df['판매수'].min())
after = df['판매수'].mean()
print('문제15',round(before-after))


# #	문제16 o
# #	vs변수가 0이 아닌 차량 중에	mpg	값이 가장 큰 차량의	hp	값을 출력하시오.
df = pd.read_csv("mtcars.csv")
print('문제16',df[~(df['vs']==0)].sort_values('mpg',ascending=False).iloc[0]['hp'])


# #	문제17 o
# #	gear변수값이3,4인 두 그룹의 hp 표준편차값의 차이를 절대값으로
# #	반올림하여 소수점 첫째자리까지 출력하시오.	
df = pd.read_csv("mtcars.csv")
g_3 = df[df['gear']==3]['hp'].std()
g_4 = df[df['gear']==4]['hp'].std()
print('문제17',round(abs(g_4-g_3),1))

# #	문제18 o 
# #	gear	변수의 값별로 그룹화하여 mpg평균값을 산출하고
# #	평균값이 높은 그룹의 mpg	제3사분위수 값을 구하시오.
df = pd.read_csv("mtcars.csv")
group = df.groupby('gear')['mpg'].mean().sort_values(ascending=False)
# print(group.head(1).index)
print('문제18',df[df['gear']==4]['mpg'].quantile(0.75))


# #	문제19 x
# #	hp	항목의 상위	7번째 값으로 상위 7개 값을 변환한 후,
# #	hp가 150 이상인 데이터를 추출하여 hp의 평균값을 반올림하여 정수로 출력하시오.
df = pd.read_csv("mtcars.csv")
df = df.reset_index(drop=True)
df['hp'].sort_values(ascending=False).iloc[6]
df['hp'] = np.where(df['hp']>=205,205,df['hp'])
# print(df['hp'].sort_values(ascending=False))
print('문제19',round(df[df['hp']>=150]['hp'].mean()))


# #	문제20 o
# #	car변수에	Merc	문구가 포함된 자동차의	mpg	평균값을 반올림하여 정수로 출력하시오
df = pd.read_csv("mtcars.csv")
print('문제20',round(df[df['car'].str.contains('Merc')]['mpg'].mean()))

# ##############################################################################################

# #	문제 21 o
# #	'22년	1분기 매출액을 구하시오
# #	(매출액	=	판매수*개당수익)
# #	데이터 생성(수정금지)
df = pd.DataFrame( {
    '날짜': ['20220103','20220105', '20230105','20230127','20220203', '20220205','20230210','20230223','20230312','20230422','20220505','20230511'],
    '제품' : ['A','B', 'A', 'B', 'A', 'B','A', 'B', 'A', 'B', 'A', 'A'],
    '판매수': [3, 5, 5, 10, 10, 10, 15, 15, 20, 25, 30, 40], 
    '개당수익': [300, 400, 500, 600, 400, 500, 500, 600, 600, 700, 600, 600] })
# print(df.info())
df['날짜'] = pd.to_datetime(df['날짜'])
df['매출액'] = df['판매수'] * df['개당수익']
con1 = df['날짜'].dt.year==2022
con2 = (df['날짜'].dt.month == 1) | (df['날짜'].dt.month == 2) | (df['날짜'].dt.month ==3)
df = df[con1 & con2]
print('문제21',df['매출액'].sum())
# 방법 2 
# df_after=df	[df['날짜'].between('2022-01-01', '2022-03-31')]
# print('문제21',df_after['매출액'].sum())


# #	문제22 o
# #	'22년과	'23년의 총 매출액 차이를 절대값으로 구하시오.	
# #	(매출액	=	판매수*개당수익)
df = pd.DataFrame( {
    '날짜': ['20220103','20220105', '20230105','20230127','20220203', '20220205','20230210','20230223','20230312','20230422','20220505','20230511'],
    '제품' : ['A','B', 'A', 'B', 'A', 'B','A', 'B', 'A', 'B', 'A', 'A'],
    '판매수': [3, 5, 5, 10, 10, 10, 15, 15, 20, 25, 30, 40], 
    '개당수익': [300, 400, 500, 600, 400, 500, 500, 600, 600, 700, 600, 600] })

df['날짜'] =pd.to_datetime(df['날짜'])
df['year'] = df['날짜'].dt.year
df['매출액'] = df['판매수'] * df['개당수익']
a = df[df['year'] == 2022]['매출액'].sum()
b = df[df['year'] == 2023]['매출액'].sum()
print('문제22',abs(a-b))



# #	문제23 x
# #	'23년 총 매출액이 큰 제품의!!!	23년 판매수를 구하시오.
# #	(매출액	=	판매수*개당수익)
df = pd.DataFrame( {
    '날짜': ['20220103','20220105', '20230105','20230127','20220203', '20220205','20230210','20230223','20230312','20230422','20220505','20230511'],
    '제품' : ['A','B', 'A', 'B', 'A', 'B','A', 'B', 'A', 'B', 'A', 'A'],
    '판매수': [3, 5, 5, 10, 10, 10, 15, 15, 20, 25, 30, 40], 
    '개당수익': [300, 400, 500, 600, 400, 500, 500, 600, 600, 700, 600, 600] })

df['날짜'] = pd.to_datetime(df['날짜'])
df['매출액'] = df['판매수'] * df['개당수익']
df['year'] = df['날짜'].dt.year
df = df[df['year']==2023]
a = df[df['제품']=='A']['매출액'].sum() # 46000
b = df[df['제품']=='B']['매출액'].sum() # 32500
print('문제23',df[df['제품']=='A']['판매수'].sum()) # 제품 A의 총 판매수



# #	문제24 o
# #	매출액이 4천원 초과, 1만원 미만인 데이터 수를 출력하시오.	
# #	(매출액	=	판매수*개당수익)
df = pd.DataFrame( {
    '날짜': ['20220103','20220105', '20230105','20230127','20220203', '20220205','20230210','20230223','20230312','20230422','20220505','20230511'],
    '제품' : ['A','B', 'A', 'B', 'A', 'B','A', 'B', 'A', 'B', 'A', 'A'],
    '판매수': [3, 5, 5, 10, 10, 10, 15, 15, 20, 25, 30, 40], 
    '개당수익': [300, 400, 500, 600, 400, 500, 500, 600, 600, 700, 600, 600] })

df['날짜'] = pd.to_datetime(df['날짜'])
df['매출액'] = df['판매수'] * df['개당수익']
print('문제24',len(df[(df['매출액']>4000) & (df['매출액']<10000)]))



#	문제25 o
#	23년 9월 24일 16:00~22:00 사이에 전체 제품의 판매수를 구하시오.	
#	시간 데이터 만들기(수정금지)
df = pd.DataFrame( {
    '물품' : ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    '판매수': [5, 10, 15, 15, 20, 25, 40], 
    '개당수익': [500, 600, 500, 600, 600, 700, 600] })
time = pd.date_range('2023-09-24 12:25:00','2023-09-25 14:45:30', periods= 7)
df['time']=time
df = df[['time','물품','판매수','개당수익']]

print('문제25',df[df['time'].between('2023-09-24 16:00:00','2023-09-24 22:00:00')]['판매수'].sum())

# #	문제26 o
# #	9월	25일	00:00~12:00	까지의	B물품의 매출액 총합을 구하시오.
# #	(매출액	=	판매수*개당수익)
# #	시간 데이터 만들기(수정금지)
df = pd.DataFrame( {
    '물품' : ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    '판매수': [5, 10, 15, 15, 20, 25, 40], 
    '개당수익': [500, 600, 500, 600, 600, 700, 600] } )
df['time'] = pd.date_range('2023-09-24 12:25:00','2023-09-25 14:45:30', periods= 7)
df = df[ ['time','물품','판매수','개당수익']]
df = df.set_index('time', drop=True)

df['매출액'] = df['판매수'] * df['개당수익']
df = df.reset_index()
df = df[df['time'].between('2023-09-25 00:00:00','2023-09-25 12:00:00')]
print('문제26',df[df['물품']=='B']['매출액'].sum())



# #	문제27 o
#	9월	24일	12:00~24:00	까지의	A물품의 매출액 총합을 구하시오.
#	(매출액	=	판매수*개당수익)
#	시간 데이터 만들기(수정금지)
df=pd.DataFrame( {
				'물품' : ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
				'판매수': [5, 10, 15, 15, 20, 25, 40],
				'개당수익': [500, 600, 500, 600, 600, 700, 600] } )
df['time'] =	pd.date_range('2023-09-24 12:25:00','2023-09-25 14:45:30',	periods= 7)
df = df[ ['time','물품','판매수','개당수익'] ]
df = df.set_index('time', drop=True)


df['매출액'] = df['판매수'] * df['개당수익']
df = df.reset_index()
df = df[df['time'].between('2023-09-24 12:00:00','2023-09-24 23:59:59')]
print('문제27',df[df['물품']=='A']['매출액'].sum())

# loc 함수로 필터링(index로 있을떄)
# df = df.loc[ (df.index >='2023-09-24 12:00:00') & (df.index <='2023-09-24	23:59:59') ]
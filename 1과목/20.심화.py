import pandas as pd
# #  menu컬럼에 "라떼" 키워드가 있는 데이터의 수는?
# sum(df['menu'].str.contains("라떼"))

# # 시간(hour)이 13시 이전(13시 포함하지 않음) 데이터 중 가장 많은 결제가 이루어진 날짜(date)는? (date 컬럼과 동일한 양식으로 출력)
# df[df['hour']<13]['date'].value_counts().index[0]

# 12월인 데이터 수는?
df = pd.read_csv('./payment.csv',parse_dates=['date'])
print(df['date'].dt.year)
df['Date'] = pd.to_datetime(df['date'],format='%y%m%d')
print(df.head())

# 수학, 영어 점수 중 사람과 과목에 상관없이 90점 이상인 점수의 평균을 정수로 구하시오 (소수점 버림)
df = pd.melt(id_vars=['name'],value_vars=['수학','영어'])
cond = df['value'] >= 90
print(int(df[cond]['value'].mean()))

# relu
def relu(x):
    return np.maximum(0, x)
df['age'] = df['age'].apply(relu)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df['f2_sigmoid'] = df['f2'].apply(sigmoid)
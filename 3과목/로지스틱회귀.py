import pandas as pd
from statsmodels.formula.api import logit
df = pd.read_csv('./data/Titanic.csv')
model = logit('Survived ~ Pclass + Gender + Embarked + SibSp ',df).fit()
print(model.summary())
# print(df.columns)
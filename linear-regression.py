import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# データ読み込み
df = pd.read_excel('2018_summer_original.xlsx',
                   sheet_name='Sheet2', encoding='utf-8')
x = df[['香港']]
y = df[['香港訪日客数']]

# 回帰線を作図
model_lr = LinearRegression()
model_lr.fit(x, y)

plt.plot(x, y, 'o')
plt.plot(x, model_lr.predict(x), linestyle="solid")
plt.show()

# 記述統計を表示
x_add_const = sm.add_constant(x)
model_sm = sm.OLS(y, x_add_const).fit()

print('モデル関数の回帰変数 w1: %.3f' % model_lr.coef_)
print('モデル関数の切片 w2: %.3f' % model_lr.intercept_)
print('y= %.3fx + %.3f' % (model_lr.coef_, model_lr.intercept_))
print('決定係数R^2:', model_lr.score(x, y))
print(model_sm.summary())

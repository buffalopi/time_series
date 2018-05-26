
# coding: utf-8

# In[1]:


#-*-coding:utf-8 -*-
#时间序列分解，上个版本做了对数转换导致 精度有所下降，同时对趋势进行预测，不适用均值
import pandas as pd
path='C:\\Users\\yang\\Desktop\\Purchase&Redemption Data\\Purchase&Redemption Data\\user_balance_table.csv'
bal=pd.read_csv(path)
summ=bal.groupby('report_date')[['total_purchase_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))


# In[131]:


#时间序列分解
import matplotlib.pyplot as plt
summ=bal.groupby('report_date')[['total_purchase_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))
#summ.total_purchase_amt=summ.total_purchase_amt.apply(lambda x:log(x))
ori=summ[summ.index>pd.to_datetime('2014-03-31')]
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ori, model="additive")
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
trend.fillna(0,inplace=True)
residual.fillna(0,inplace=True)
trend.plot(figsize=(12,8))
seasonal.plot(figsize=(12,8))
residual.plot(figsize=(12,8))
plt.show()


# In[132]:


#趋势分析
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
trend_s=pd.Series(list(trend.total_purchase_amt),index=list(trend.index))
trend_s=trend_s[3:]
trend_s=trend_s[:-3]
adfuller(trend_s)
plot_acf(trend_s,lags=50)
plot_pacf(trend_s,lags=50)
plt.show()
diff=trend_s.diff().fillna(0)
diff.plot()
plt.show()
plot_acf(diff,lags=50)
plot_pacf(diff,lags=50)
plt.show()


# In[133]:


#趋势建模
from statsmodels.tsa.arima_model import ARIMA,ARMA
model=ARMA(diff,order=(7,6)).fit()
predict=model.predict(0,179)
resids=model.resid
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(resids))
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(resids,lags=1)


# In[134]:


#趋势结果
import numpy as np
it_p=predict[-33:].values
his=[]
x=trend_s[trend_s.index==pd.to_datetime('2014-08-28')].values[0]
his.append(x)
for i in it_p:
    his.append(his[-1]+i)
his=np.array(his[4:])
#残差 分析建模
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(residual,lags=50)
plot_pacf(residual,lags=50)
plt.show()
from statsmodels.tsa.stattools import adfuller
res=pd.Series(list(residual.total_purchase_amt),index=list(seasonal.index))
adfuller(res)
from statsmodels.tsa.arima_model import ARIMA,ARMA
model=ARMA(res,order=(2,4)).fit()
predict=model.predict(153,182)
resids=model.resid
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(resids))
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(resids,lags=1)


# In[135]:





# In[136]:


#周期性
import numpy as np
sea=np.array((seasonal.tail(35)[0:30].total_purchase_amt.values))
#求和
p=np.array(predict.values)
purchase=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
purchase['p']=sea+p+his


# In[140]:


#redeem
#时间序列分解
import matplotlib.pyplot as plt
summ=bal.groupby('report_date')[['total_redeem_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))
#summ.total_purchase_amt=summ.total_purchase_amt.apply(lambda x:log(x))
ori=summ[summ.index>pd.to_datetime('2014-03-31')]
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ori, model="additive")
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
trend.fillna(0,inplace=True)
residual.fillna(0,inplace=True)
#trend.plot(figsize=(12,8))
#seasonal.plot(figsize=(12,8))
#residual.plot(figsize=(12,8))
#plt.show()


# In[142]:


#趋势分析
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
trend_s=pd.Series(list(trend.total_redeem_amt),index=list(trend.index))
trend_s=trend_s[3:]
trend_s=trend_s[:-3]
adfuller(trend_s)
plot_acf(trend_s,lags=50)
plot_pacf(trend_s,lags=50)
plt.show()
diff=trend_s.diff().fillna(0)
diff.plot()
plt.show()
plot_acf(diff,lags=50)
plot_pacf(diff,lags=50)
plt.show()
#趋势建模
from statsmodels.tsa.arima_model import ARIMA,ARMA
model=ARMA(diff,order=(7,6)).fit()
predict=model.predict(0,179)
resids=model.resid
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(resids))
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(resids,lags=1)
#趋势
import numpy as np
it_p=predict[-33:].values
his=[]
x=trend_s[trend_s.index==pd.to_datetime('2014-08-28')].values[0]
his.append(x)
for i in it_p:
    his.append(his[-1]+i)
his=np.array(his[4:])
#残差 分析建模
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(residual,lags=50)
plot_pacf(residual,lags=50)
plt.show()
from statsmodels.tsa.stattools import adfuller
res=pd.Series(list(residual.total_redeem_amt),index=list(seasonal.index))
adfuller(res)
from statsmodels.tsa.arima_model import ARIMA,ARMA
model=ARMA(res,order=(2,4)).fit()
predict=model.predict(153,182)
resids=model.resid
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(resids))
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(resids,lags=1)
#周期性
import numpy as np
sea=np.array((seasonal.tail(35)[0:30].total_redeem_amt.values))
#求和
p=np.array(predict.values)
red=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
red['p']=sea+p+his


# In[152]:





# In[163]:


purchase['red']=red.p
purchase.to_csv('G:\\tc_comp_predict_table.csv',header=False)


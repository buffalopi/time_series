
# coding: utf-8

# In[1]:


#-*-coding:utf-8 -*-
import pandas as pd
path='C:\\Users\\yang\\Desktop\\Purchase&Redemption Data\\Purchase&Redemption Data\\user_balance_table.csv'
bal=pd.read_csv(path)
summ=bal.groupby('report_date')[['total_purchase_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))


# In[5]:


from math import log
import matplotlib.pyplot as plt
summ=bal.groupby('report_date')[['total_purchase_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))
summ.total_purchase_amt=summ.total_purchase_amt.apply(lambda x:log(x))
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
#检验
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
#周期性
import numpy as np
sea=np.array((seasonal.tail(35)[0:30].total_purchase_amt.values))
#趋势
mean_trend=np.mean(trend.tail(60).head(57)).values[0]
p=np.array(predict.values)
purchase=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
purchase['p']=sea+p+mean_trend
from math import exp
purchase.p=purchase.p.apply(lambda x:exp(x))


# In[74]:


from math import log
import matplotlib.pyplot as plt
summ=bal.groupby('report_date')[['total_redeem_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))
summ.total_redeem_amt=summ.total_redeem_amt.apply(lambda x:log(x))
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


# In[83]:



#检验
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(residual,lags=50)
plot_pacf(residual,lags=50)
plt.show()
from statsmodels.tsa.stattools import adfuller
res=pd.Series(list(residual.total_redeem_amt),index=list(seasonal.index))
adfuller(res)
from statsmodels.tsa.arima_model import ARIMA,ARMA
model=ARMA(res,order=(6,3)).fit()
predict=model.predict(153,182)
resids=model.resid
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(resids))
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(resids,lags=1)


# In[85]:


#周期性
import numpy as np
sea=np.array((seasonal.tail(35)[0:30].total_redeem_amt.values))
#趋势
mean_trend=np.mean(trend.tail(60).head(57)).values[0]
p=np.array(predict.values)
redeem=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
redeem['p']=sea+p+mean_trend
from math import exp
redeem.p=redeem.p.apply(lambda x:exp(x))


# In[87]:


purchase['red']=redeem.p


# In[89]:


purchase.to_csv('G:\\tc_comp_predict_table.csv',header=False)


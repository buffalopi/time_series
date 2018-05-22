
# coding: utf-8

# In[2]:


#天池流入流出预测
#用户信息
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:

path='G:\\Purchase&Redemption Data\\Purchase&Redemption Data\\user_profile_table.csv'
user=pd.read_csv(path)


# In[6]:

user.head()
#性别
#城市
#星座


# In[22]:

user.isnull().sum()


# In[10]:

user.size


# In[13]:

#检测数据是否重复
user.drop_duplicates(inplace=True)
user.size


# In[9]:

user.sex.value_counts().plot(kind='bar')
plt.show()


# In[15]:

user.city.value_counts().plot(kind='bar')
plt.show()


# In[18]:

user.constellation.value_counts().plot(kind='bar')
plt.show()


# In[19]:

#资金流水表
path='G:\\Purchase&Redemption Data\\Purchase&Redemption Data\\user_balance_table.csv'
bal=pd.read_csv(path)


# In[20]:

bal.head()


# In[23]:

bal.isnull().sum()


# In[24]:

bal.size


# In[26]:

#检测是否有重复数据
bal.drop_duplicates(inplace=True)
bal.size


# In[29]:

bal.drop(['category1','category2','category3','category4'],inplace=True,axis=1)


# In[32]:

bal[bal.user_id==1].sort_values('report_date')
#今日余额=昨日余额+今日总购买量
#今日总购买量=今日收益+直接购买


# In[43]:

#汇总每日的申购总额
purchase=bal.groupby('report_date')['total_purchase_amt'].sum()


# In[45]:

#时间范围
purchase.index


# In[56]:

purchase.index=pd.to_datetime(purchase.index.map(str))


# In[58]:

purchase.plot(figsize=(12,8))
plt.show()


# In[167]:

#选取4月份以后的数据,原因是4月份以后的数据比较平滑,目标是预测9月份数据,与较长历史数据关系不大
sub_pur=purchase[purchase.index>pd.to_datetime('2014-03-31')]
sub_pur.plot(figsize=(12,8))
plt.show()
#数据有很明显的周的周期性


# In[61]:

#检查数据是否平稳,单方根,原假设数据是不平稳的
from statsmodels.tsa.stattools import adfuller
adfuller(sub_pur)


# In[63]:

diff=sub_pur.diff(1).fillna(0)
adfuller(diff)
#已经是平稳的序列


# In[64]:

#acf,pacf 定阶
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(diff,lags=50)
plot_pacf(diff,lags=50)
plt.show()
#p,q 遍历1~6


# In[133]:

#选取模型
#A forecasting model with a unit root in the estimated MA coefficients is said to be noninvertible, 
#meaning that the residuals of the model cannot be considered as estimates of the "true" random noise that generated the time series.
aic=[]
bic=[]
hqic=[]
pl=[]
ql=[]
from statsmodels.tsa.arima_model import ARIMA,ARMA
for p in range(1,8):
    for q in range(1,8):
        try:
            model=ARMA(diff,order=(p,q)).fit()
            print('model  p is %d ,q is %d '%(p,q))
            print('aic is %f ,bic is %f ,hqic is %f',(model.aic,model.bic,model.hqic))
            aic.append(model.aic)
            bic.append(model.bic)
            hqic.append(model.hqic)
            pl.append(p)
            ql.append(q)
        except:
            print('不合理的p,q组合')
                


# In[134]:

ic=pd.DataFrame()
ic['p']=pl
ic['q']=ql
ic['aic']=aic
ic['bic']=bic
ic['hqic']=hqic
ic


# In[122]:

#定阶arma(3,5),arma(4,5),arma(6,5)
#白噪声检验
model_35=ARIMA(diff,order=(3,0,5)).fit()
red_35=model_35.resid
model_45=ARIMA(diff,order=(4,0,5)).fit()
red_45=model_45.resid
model_65=ARIMA(diff,order=(6,0,5)).fit()
red_65=model_65.resid


# In[124]:

plot_acf(red_35)
plot_pacf(red_35)
plot_acf(red_45)
plot_pacf(red_45)
plot_acf(red_65)
plot_pacf(red_65)
plt.show()


# In[125]:

#D-W德宾-沃森检验 自相关性
#arma(3,5)最好
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(red_35))
print(durbin_watson(red_45))
print(durbin_watson(red_65))


# In[128]:

red_35.plot(figsize=(12,8))
plt.show()


# In[132]:

#原假设是白噪音
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(red_35,lags=1)
acorr_ljungbox(sub_pur,lags=1)


# In[138]:

#自动选择,p,q的阶数
from statsmodels.tsa.stattools import arma_order_select_ic
arma_order_select_ic(diff,max_ar=7,max_ma=7,ic='aic')['aic_min_order']


# In[139]:

arma_order_select_ic(diff,max_ar=7,max_ma=7,ic='bic')['bic_min_order']


# In[140]:

arma_order_select_ic(diff,max_ar=7,max_ma=7,ic='hqic')['hqic_min_order']


# In[142]:

#拒绝原假设,残差不是白噪声
model_23=ARIMA(diff,order=(2,0,3)).fit()
red_23=model_23.resid
acorr_ljungbox(red_23,lags=1)


# In[197]:

#预测
pred=model_35.predict(0,152,dynamic=False)
diff.plot()
re=diff.reset_index()
re['pred']=pred.values
re.set_index('report_date')
re.plot(figsize=(12,8))
plt.show()


# In[210]:

#predict,预测差分效果对比
x=[]
x.append(sub_pur[0])
for i in diff.values:
    x.append(x[-1]+i)
print(x[1:4])
print(sub_pur[0:3])


# In[361]:

#模型1 结果
pred=model_35.predict(153,182,dynamic=False)
x=[]
x.append(sub_pur[-1])
for i in pred.values:
    x.append(x[-1]+i)
y=x[1:]
predict=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
predict['total_purchase_amt']=y
predict.head()


# In[238]:

#滚动预测 predict forecast 区别 有出入但整体不大
mul_steps_diff=model_35.forecast(steps=30)[0]


# In[311]:

#
diff_cp=diff
for i in range(1,31):
    model_35=ARIMA(diff_cp,order=(3,0,5)).fit()
    pt=model_35.forecast()[0]
    idx=pd.to_datetime('9/'+str(i)+'/2014')
    diff_cp=diff_cp.append(pd.Series(pt,index=[idx]),ignore_index=False)


# In[318]:

roll=diff_cp[diff_cp.index>pd.to_datetime('2014-08-31')]
df=roll.reset_index()
df['mul']=mul_steps_diff
df.set_index('index')
df.plot(figsize=(12,8))
plt.show()


# In[329]:

#赎回预测redeem
redeem=bal.groupby('report_date')['total_redeem_amt'].sum()


# In[320]:

redeem.head()


# In[331]:

redeem.index=pd.to_datetime(redeem.index.map(str))
redeem.plot(figsize=(12,8))
plt.show()


# In[333]:

sub_red=redeem[redeem.index>pd.to_datetime('2014-03-31')]
sub_red.plot(figsize=(12,8))
plt.show()


# In[334]:

#平稳性检验,存在单位根
from statsmodels.stats.diagnostic import acorr_ljungbox
adfuller(sub_red)


# In[335]:

red_diff=sub_red.diff(1).fillna(0)
adfuller(red_diff)


# In[338]:

red_diff.plot(figsize=(12,8))
plot_acf(red_diff,lags=50)
plot_pacf(red_diff,lags=50)
plt.show()


# In[352]:

#定阶
#选取模型
#A forecasting model with a unit root in the estimated MA coefficients is said to be noninvertible, 
#meaning that the residuals of the model cannot be considered as estimates of the "true" random noise that generated the time series.
aic=[]
bic=[]
hqic=[]
pl=[]
ql=[]
from statsmodels.tsa.arima_model import ARIMA,ARMA
for p in range(1,8):
    for q in range(1,8):
        try:
            model=ARIMA(red_diff,order=(p,0,q)).fit()
            print('model  p is %d ,q is %d '%(p,q))
            print('aic is %f ,bic is %f ,hqic is %f',(model.aic,model.bic,model.hqic))
            aic.append(model.aic)
            bic.append(model.bic)
            hqic.append(model.hqic)
            pl.append(p)
            ql.append(q)
        except:
            print('不合理的p,q组合')


# In[353]:

ic=pd.DataFrame()
ic['p']=pl
ic['q']=ql
ic['aic']=aic
ic['bic']=bic
ic['hqic']=hqic
ic


# In[354]:

#arma(6,7),(2,4)
#d-w检验,(2,4)效果最好
model_67=ARIMA(red_diff,order=(6,0,7)).fit()
red_67=model_67.resid
model_24=ARIMA(red_diff,order=(2,0,4)).fit()
red_24=model_24.resid
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(red_67))
print(durbin_watson(red_24))


# In[355]:

plot_acf(red_24)
plot_pacf(red_24)
plt.show()


# In[359]:

redeem_model=ARIMA(red_diff,order=(2,0,4)).fit()
results=redeem_model.forecast(steps=30)[0]
x=[]
x.append(red_diff[-1])
for i in results:
    x.append(x[-1]+i)
y=x[1:]
red_pre=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
red_pre['total_redeem_amt']=y
red_pre.head()


# In[375]:

version_1=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
version_1['purchase']=predict.total_purchase_amt
version_1['redeem']=red_pre.total_redeem_amt
version_1.reset_index(inplace=True)
version_1.to_csv('G:\\tc_comp_predict_table.csv',index=False,header=False)


# In[351]:

#预测8月份结果
train,test=diff[diff.index<pd.to_datetime('2014-08-01')],diff[diff.index>pd.to_datetime('2014-07-31')]
model_test=ARIMA(train,order=(3,0,5)).fit()
predict=model_test.forecast(steps=31)[0]
x=[]
x.append(train[-1])
for i in predict:
    x.append(x[-1]+i)
y=x[1:]
df=test.reset_index()
df['pred']=y
df.set_index('report_date',inplace=True)
df.plot(figsize=(12,8))
plt.show()


# In[408]:

#AR(3)
from statsmodels.tsa.ar_model import AR
ar_3=AR(diff).fit(ic='hqic')
ar_3_pre=ar_3.predict()


# In[411]:

diff_copy=diff[diff.index>pd.to_datetime('2014-04-13')]
ardf=diff_copy.reset_index()
ardf['pred']=ar_3_pre.values
ardf.set_index('report_date',inplace=True)
ardf.plot(figsize=(12,8))
plt.show()


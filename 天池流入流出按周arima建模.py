
# coding: utf-8

# In[2]:


#天池资金流入流出预测版本2 
#对时间序列,按星期分类,做ARIMA预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox


# In[18]:

#资金流水表
path='G:\\Purchase&Redemption Data\\Purchase&Redemption Data\\user_balance_table.csv'
bal=pd.read_csv(path)
#汇总每日的申购总额
purchase=bal.groupby('report_date')['total_purchase_amt'].sum()
purchase.index=pd.to_datetime(purchase.index.map(str))
sub_pur=purchase[purchase.index>pd.to_datetime('2014-03-31')]
sub_pur_cp=sub_pur.reset_index()
week=list(sub_pur.index.dayofweek)
sub_pur_cp['week']=week
purchase_1=sub_pur_cp[sub_pur_cp.week==0]
purchase_2=sub_pur_cp[sub_pur_cp.week==1]
purchase_3=sub_pur_cp[sub_pur_cp.week==2]
purchase_4=sub_pur_cp[sub_pur_cp.week==3]
purchase_5=sub_pur_cp[sub_pur_cp.week==4]
purchase_6=sub_pur_cp[sub_pur_cp.week==5]
purchase_7=sub_pur_cp[sub_pur_cp.week==6]
#按周划分数据
pur_ts_1=pd.Series(list(purchase_1.total_purchase_amt),index=purchase_1.report_date) 
pur_ts_2=pd.Series(list(purchase_2.total_purchase_amt),index=purchase_2.report_date) 
pur_ts_3=pd.Series(list(purchase_3.total_purchase_amt),index=purchase_3.report_date) 
pur_ts_4=pd.Series(list(purchase_4.total_purchase_amt),index=purchase_4.report_date) 
pur_ts_5=pd.Series(list(purchase_5.total_purchase_amt),index=purchase_5.report_date) 
pur_ts_6=pd.Series(list(purchase_6.total_purchase_amt),index=purchase_6.report_date) 
pur_ts_7=pd.Series(list(purchase_7.total_purchase_amt),index=purchase_7.report_date)
#差分
pur_diff=[]
for i in range(1,8):
    pur_diff.append(eval('pur_ts_'+str(i)).diff(1).fillna(0))
#周一,残差是白噪声
week_1=ARIMA(pur_diff[0],order=(1,0,1)).fit()
print(durbin_watson(week_1.resid))
print(acorr_ljungbox(week_1.resid,lags=1))
pred_1=week_1.forecast(steps=5)[0]
his_1=[]
his_1.append(pur_ts_1[-1])
for i in pred_1:
    his_1.append(his_1[-1]+i)
y_1=his_1[1:]
#周二,残差是白噪声
week_2=ARIMA(pur_diff[1],order=(2,0,1)).fit()
print(durbin_watson(week_2.resid))
print(acorr_ljungbox(week_2.resid,lags=1))
pred_2=week_2.forecast(steps=5)[0]
his_2=[]
his_2.append(pur_ts_2[-1])
for i in pred_2:
    his_2.append(his_2[-1]+i)
y_2=his_2[1:]
#周三,残差是白噪声
week_3=ARIMA(pur_diff[2],order=(1,0,2)).fit()
print(durbin_watson(week_3.resid))
print(acorr_ljungbox(week_3.resid,lags=1))
pred_3=week_3.forecast(steps=5)[0]
his_3=[]
his_3.append(pur_ts_3[-1])
for i in pred_3:
    his_3.append(his_3[-1]+i)
y_3=his_3[1:]
#周四,残差是白噪声
week_4=ARIMA(pur_diff[3],order=(1,0,1)).fit()
print(durbin_watson(week_4.resid))
print(acorr_ljungbox(week_4.resid,lags=1))
pred_4=week_4.forecast(steps=5)[0]
his_4=[]
his_4.append(pur_ts_4[-1])
for i in pred_4:
    his_4.append(his_4[-1]+i)
y_4=his_4[1:]
#周五,残差是白噪声
week_5=ARIMA(pur_diff[4],order=(2,0,1)).fit()
print(durbin_watson(week_5.resid))
print(acorr_ljungbox(week_5.resid,lags=1))
pred_5=week_5.forecast(steps=5)[0]
his_5=[]
his_5.append(pur_ts_5[-1])
for i in pred_5:
    his_5.append(his_5[-1]+i)
y_5=his_5[1:]
#周六,残差是白噪声
week_6=ARIMA(pur_diff[5],order=(2,0,1)).fit()
print(durbin_watson(week_6.resid))
print(acorr_ljungbox(week_6.resid,lags=1))
pred_6=week_6.forecast(steps=5)[0]
his_6=[]
his_6.append(pur_ts_6[-1])
for i in pred_6:
    his_6.append(his_6[-1]+i)
y_6=his_6[1:]
#周日,残差是白噪声
week_7=ARIMA(pur_diff[6],order=(2,0,1)).fit()
print(durbin_watson(week_7.resid))
print(acorr_ljungbox(week_7.resid,lags=1))
pred_7=week_7.forecast(steps=5)[0]
his_7=[]
his_7.append(pur_ts_7[-1])
for i in pred_7:
    his_7.append(his_7[-1]+i)
y_7=his_7[1:]
print(sub_pur_cp.tail())
purchase_predict=[]
for i in range(0,5):
    purchase_predict.append(y_7[i])
    purchase_predict.append(y_1[i])
    purchase_predict.append(y_2[i])
    purchase_predict.append(y_3[i])
    purchase_predict.append(y_4[i])
    purchase_predict.append(y_5[i])
    purchase_predict.append(y_6[i])
pred_8=purchase_predict[:30]
pred_9=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
pred_9['total_purchase_amt']=pred_8
pred_9.plot()
plt.show()


# In[23]:

#赎回的预测
#汇总每日的申购总额
purchase=bal.groupby('report_date')['total_redeem_amt'].sum()
purchase.index=pd.to_datetime(purchase.index.map(str))
sub_pur=purchase[purchase.index>pd.to_datetime('2014-03-31')]
sub_pur_cp=sub_pur.reset_index()
week=list(sub_pur.index.dayofweek)
sub_pur_cp['week']=week
purchase_1=sub_pur_cp[sub_pur_cp.week==0]
purchase_2=sub_pur_cp[sub_pur_cp.week==1]
purchase_3=sub_pur_cp[sub_pur_cp.week==2]
purchase_4=sub_pur_cp[sub_pur_cp.week==3]
purchase_5=sub_pur_cp[sub_pur_cp.week==4]
purchase_6=sub_pur_cp[sub_pur_cp.week==5]
purchase_7=sub_pur_cp[sub_pur_cp.week==6]
#按周划分数据
pur_ts_1=pd.Series(list(purchase_1.total_redeem_amt),index=purchase_1.report_date) 
pur_ts_2=pd.Series(list(purchase_2.total_redeem_amt),index=purchase_2.report_date) 
pur_ts_3=pd.Series(list(purchase_3.total_redeem_amt),index=purchase_3.report_date) 
pur_ts_4=pd.Series(list(purchase_4.total_redeem_amt),index=purchase_4.report_date) 
pur_ts_5=pd.Series(list(purchase_5.total_redeem_amt),index=purchase_5.report_date) 
pur_ts_6=pd.Series(list(purchase_6.total_redeem_amt),index=purchase_6.report_date) 
pur_ts_7=pd.Series(list(purchase_7.total_redeem_amt),index=purchase_7.report_date)
#差分
pur_diff=[]
for i in range(1,8):
    pur_diff.append(eval('pur_ts_'+str(i)).diff(1).fillna(0))
    
for i in range(0,7):
    print('周'+str(i+1)+'的 acf pacf 图')
    plot_acf(pur_diff[i])
    plot_pacf(pur_diff[i])
    plt.show()
#周一,残差是白噪声
week_1=ARIMA(pur_diff[0],order=(3,0,2)).fit()
print(durbin_watson(week_1.resid))
print(acorr_ljungbox(week_1.resid,lags=1))
pred_1=week_1.forecast(steps=5)[0]
his_1=[]
his_1.append(pur_ts_1[-1])
for i in pred_1:
    his_1.append(his_1[-1]+i)
y_1=his_1[1:]
#周二,残差是白噪声
week_2=ARIMA(pur_diff[1],order=(2,0,1)).fit()
print(durbin_watson(week_2.resid))
print(acorr_ljungbox(week_2.resid,lags=1))
pred_2=week_2.forecast(steps=5)[0]
his_2=[]
his_2.append(pur_ts_2[-1])
for i in pred_2:
    his_2.append(his_2[-1]+i)
y_2=his_2[1:]
#周三,残差是白噪声
week_3=ARIMA(pur_diff[2],order=(3,0,2)).fit()
print(durbin_watson(week_3.resid))
print(acorr_ljungbox(week_3.resid,lags=1))
pred_3=week_3.forecast(steps=5)[0]
his_3=[]
his_3.append(pur_ts_3[-1])
for i in pred_3:
    his_3.append(his_3[-1]+i)
y_3=his_3[1:]
#周四,残差是白噪声
week_4=ARIMA(pur_diff[3],order=(3,0,2)).fit()
print(durbin_watson(week_4.resid))
print(acorr_ljungbox(week_4.resid,lags=1))
pred_4=week_4.forecast(steps=5)[0]
his_4=[]
his_4.append(pur_ts_4[-1])
for i in pred_4:
    his_4.append(his_4[-1]+i)
y_4=his_4[1:]
#周五,残差是白噪声
week_5=ARIMA(pur_diff[4],order=(5,0,1)).fit()
print(durbin_watson(week_5.resid))
print(acorr_ljungbox(week_5.resid,lags=1))
pred_5=week_5.forecast(steps=5)[0]
his_5=[]
his_5.append(pur_ts_5[-1])
for i in pred_5:
    his_5.append(his_5[-1]+i)
y_5=his_5[1:]
#周六,残差是白噪声
week_6=ARIMA(pur_diff[5],order=(2,0,3)).fit()
print(durbin_watson(week_6.resid))
print(acorr_ljungbox(week_6.resid,lags=1))
pred_6=week_6.forecast(steps=5)[0]
his_6=[]
his_6.append(pur_ts_6[-1])
for i in pred_6:
    his_6.append(his_6[-1]+i)
y_6=his_6[1:]
#周日,残差是白噪声
week_7=ARIMA(pur_diff[6],order=(6,0,5)).fit()
print(durbin_watson(week_7.resid))
print(acorr_ljungbox(week_7.resid,lags=1))
pred_7=week_7.forecast(steps=5)[0]
his_7=[]
his_7.append(pur_ts_7[-1])
for i in pred_7:
    his_7.append(his_7[-1]+i)
y_7=his_7[1:]
print(sub_pur_cp.tail())
purchase_predict=[]
for i in range(0,5):
    purchase_predict.append(y_7[i])
    purchase_predict.append(y_1[i])
    purchase_predict.append(y_2[i])
    purchase_predict.append(y_3[i])
    purchase_predict.append(y_4[i])
    purchase_predict.append(y_5[i])
    purchase_predict.append(y_6[i])
pred_red=purchase_predict[:30]
pred_0=pd.DataFrame(index=pd.date_range(start='2014-09-01',end='2014-09-30'))
pred_0['total_redeem_amt']=pred_red
pred_0.plot()
plt.show()


# In[47]:

#8月9月数据对比图
comp=purchase[purchase.index>pd.to_datetime('2014-07-31')]
comp=pd.DataFrame(comp.values,index=comp.index,columns=['total_redeem_amt'])
df=pd.concat([comp,pred_0],axis=0)
df.plot(figsize=(12,8))
plt.show()


# In[53]:

#汇总数据保存
df=pred_9
df['total_redeem_amt']=pred_red
#pth='G:\\tc_comp_predict_table.csv'
#v1=pd.read_csv(pth,header=None)
#df['p_a']=list(v1.iloc[:,1])
#df['red_b']=list(v1.iloc[:,2])
#df.plot(figsize=(12,8))
#plt.show()
pred_9.drop(['p_a','red_b'],axis=1,inplace=True)


# In[56]:

pred_9.to_csv('G:\\tc_comp_predict_table.csv',header=False)


# In[59]:

f=bal.groupby('report_date')['total_redeem_amt','total_purchase_amt'].sum()
f.index=pd.to_datetime(f.index.map(str))
f[f.index>pd.to_datetime('2014-04-01')].plot(figsize=(12,8))
plt.show()


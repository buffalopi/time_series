
# coding: utf-8

# In[2]:


#-*- coding:utf-8 -*-
import pandas as pd
path='C:\\Users\\yang\\Desktop\\Purchase&Redemption Data\\Purchase&Redemption Data\\user_balance_table.csv'
bal=pd.read_csv(path)


# In[60]:


summ=bal.groupby('report_date')[['total_purchase_amt','total_redeem_amt']].sum()
summ.index=pd.to_datetime(summ.index.map(str))
#前7天数据
lag_1=summ[(summ.index<pd.to_datetime('2014-08-31'))&((summ.index>=pd.to_datetime('2014-03-31')))].values
lag_2=summ[(summ.index<pd.to_datetime('2014-08-30'))&((summ.index>=pd.to_datetime('2014-03-30')))].values
lag_3=summ[(summ.index<pd.to_datetime('2014-08-29'))&((summ.index>=pd.to_datetime('2014-03-29')))].values
lag_4=summ[(summ.index<pd.to_datetime('2014-08-28'))&((summ.index>=pd.to_datetime('2014-03-28')))].values
lag_5=summ[(summ.index<pd.to_datetime('2014-08-27'))&((summ.index>=pd.to_datetime('2014-03-27')))].values
lag_6=summ[(summ.index<pd.to_datetime('2014-08-26'))&((summ.index>=pd.to_datetime('2014-03-26')))].values
lag_7=summ[(summ.index<pd.to_datetime('2014-08-25'))&((summ.index>=pd.to_datetime('2014-03-25')))].values
sub=summ[summ.index>pd.to_datetime('2014-03-31')]


# In[104]:


sub_pur=sub.drop('total_redeem_amt',axis=1)
sub_pur['lag_1']=lag_1[:,0]
sub_pur['lag_2']=lag_2[:,0]
sub_pur['lag_3']=lag_3[:,0]
sub_pur['lag_4']=lag_4[:,0]
sub_pur['lag_5']=lag_5[:,0]
sub_pur['lag_6']=lag_6[:,0]
#sub_pur['lag_7']=lag_7[:,0]


# In[105]:


import numpy as np
#sub=summ[summ.index>pd.to_datetime('2014-03-31')]
week=np.array(sub_pur.index.dayofweek)
sub_pur['week']=week
sub_week=pd.get_dummies(sub_pur,columns=['week'])


# In[73]:


sub_week.head()


# In[107]:


train=sub_week[sub_week.index<pd.to_datetime('2014-08-01')]
test=trian=sub_week[sub_week.index>=pd.to_datetime('2014-08-01')]
X_train=train.iloc[:,1:]
y_train=train.iloc[:,0]
X_test=test.iloc[:,1:]
y_test=test.iloc[:,0]


# In[108]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)


# In[103]:


df=pd.DataFrame(y_test)
df['pred']=pred
import matplotlib.pyplot as plt
df.plot(figsize=(12,8))
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test.values,pred)


# In[109]:


df=pd.DataFrame(y_test)
df['pred']=pred
import matplotlib.pyplot as plt
df.plot(figsize=(12,8))
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test.values,pred)


# In[151]:


#9 mon
X_train=sub_week.iloc[:,1:]
y_train=sub_week.iloc[:,0]
his=list(sub_week['total_purchase_amt'].values[-10:])
lr=LinearRegression()
lr.fit(X_train,y_train)
x=pd.DataFrame(index=pd.date_range('2014-09-01','2014-09-30'))
x['week']=x.index.dayofweek
x=pd.get_dummies(x,columns=['week'])


# In[152]:


for i in range(1,31):
    test=x[x.index==pd.to_datetime('9/'+str(i)+'/2014')]
    test['lag_1']=his[-1]
    test['lag_2']=his[-2]
    test['lag_3']=his[-3]
    test['lag_4']=his[-4]
    test['lag_5']=his[-5]
    test['lag_6']=his[-6]
    test=test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6','week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6'
       ]]
    y=lr.predict(test)[0]
    his.append(y)


# In[162]:


pur=his[10:]
vie=pd.DataFrame(index=pd.date_range('2014-09-01','2014-09-30'))


# In[165]:


#redeem
sub_pur=sub.drop('total_purchase_amt',axis=1)
sub_pur['lag_1']=lag_1[:,0]
sub_pur['lag_2']=lag_2[:,0]
sub_pur['lag_3']=lag_3[:,0]
sub_pur['lag_4']=lag_4[:,0]
sub_pur['lag_5']=lag_5[:,0]
sub_pur['lag_6']=lag_6[:,0]
#sub_pur['lag_7']=lag_7[:,0]
import numpy as np
#sub=summ[summ.index>pd.to_datetime('2014-03-31')]
week=np.array(sub_pur.index.dayofweek)
sub_pur['week']=week
sub_week=pd.get_dummies(sub_pur,columns=['week'])

X_train=sub_week.iloc[:,1:]
y_train=sub_week.iloc[:,0]
his=list(sub_week['total_redeem_amt'].values[-10:])
lr=LinearRegression()
lr.fit(X_train,y_train)
x=pd.DataFrame(index=pd.date_range('2014-09-01','2014-09-30'))
x['week']=x.index.dayofweek
x=pd.get_dummies(x,columns=['week'])
for i in range(1,31):
    test=x[x.index==pd.to_datetime('9/'+str(i)+'/2014')]
    test['lag_1']=his[-1]
    test['lag_2']=his[-2]
    test['lag_3']=his[-3]
    test['lag_4']=his[-4]
    test['lag_5']=his[-5]
    test['lag_6']=his[-6]
    test=test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6','week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6'
       ]]
    y=lr.predict(test)[0]
    his.append(y)


# In[167]:


red=his[10:]
vie['pur']=pur
vie['red']=red


# In[169]:


vie.plot()
plt.show()


# In[170]:


nt=summ[summ.index>pd.to_datetime('2014-07-30')]
nt.plot()
plt.show()


# In[175]:


vie.to_csv('F:\\tc_comp_predict_table.csv',header=False)


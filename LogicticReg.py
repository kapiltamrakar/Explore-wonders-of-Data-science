#!/usr/bin/env python
# coding: utf-8

# In[191]:


#in this logistic regression, we are trying to predict in case customer buys the FD from bank (target variable y=1) 
# or not (target variable y=0)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\\ktamrakar\\Desktop\\Datascience\\Logictic regression\\Banking.csv")
#checking the first 5 records, to confirm is data is ready properly. 
df.head()


# In[42]:


#checking # of columns and rows in the data set
df.shape
# it has 41188 rows and 21 columns


# In[10]:


#checking if any column has null value
df.info()


# In[43]:


#seems no null value in any column
#now we will try to see if categorical variable has impact on target variable(y)
#lets see Education
df["education"].unique()


# In[44]:


#Lets group  basic.4y, basic.6y and basic.9y to Basic_edu so that we will have less categorical values to analyze
df["education"] = np.where(df["education"]=="basic.4y", "Basic_edu", df["education"])
df["education"] = np.where(df["education"]=="basic.6y", "Basic_edu", df["education"])
df["education"] = np.where(df["education"]=="basic.9y", "Basic_edu", df["education"])


# In[45]:


#checking again the columns
df["education"].unique()


# In[46]:


#its showing the Basic_Edu which is combinition of basic.4y, basic.6y, basic.9y
#lets explore more data, to see how many customer have purchased the FD
df["y"].value_counts()


# In[30]:


#it seems only 4640 customer have bought the FD and other 36548 didnt. also we can see its imbalance data set.
sns.countplot(x='y',data=df, palette='hls')
plt.show()


# In[63]:


#seeing the same in count plot
#check the % of Positive outcome and Negatove count come in output variable (y)
count_of_positive_outcome = len(df[df["y"]==1])
count_of_negative_outcome = len(df[df["y"]==0])
Per_positive_outcome = (count_of_positive_outcome / (count_of_negative_outcome+count_of_positive_outcome)) * 100
Per_negative_outcome = (count_of_negative_outcome / (count_of_negative_outcome+count_of_positive_outcome)) * 100
print("customer baught FD %", Per_positive_outcome)
print("customer didnt buy FD %", Per_negative_outcome)


# In[64]:


# so we can see this data set is imbalanec and if we train our model in this data set, it could be skewd towards "customer didnt buy FD" outcome 
# lets do some more reasearch
#lets try to see stats across Positive(customer baught FD)and negative outcome (customer didnt buy FD)
df.groupby("y").mean()


# In[68]:


# above is the mean value of all the numeric columns, we can see, avg age of costomer who baught FD is more than who didnt. 
# So it seems age is a the factor influencing the decision
# now lets analyze the data visually
#plotting bar graph for JOb and y
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.job, df.y).plot(kind="bar")
plt.title("Purchase of frequency vs Job")
plt.xlabel("Job")
plt.ylabel("Frequence of purchase")


# In[69]:


#we can see, admin category job have purchased FD more. 
#lets try now with marital status
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.marital, df.y).plot(kind="bar")
plt.title("Purchase of frequency vs marital status")
plt.xlabel("Marital status")
plt.ylabel("Frequence of purchase")


# In[70]:


# we can see, marital and single status customer purchansed FD. 
#lets try now with Education
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.education, df.y).plot(kind="bar")
plt.title("Purchase of frequency vs education status")
plt.xlabel("Education")
plt.ylabel("Frequence of purchase")


# In[71]:


#seems educated customer have purchansed FD
#lets see which day of the week, customer have purchased FD

get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.day_of_week, df.y).plot(kind="bar")
plt.title("Purchase of frequency vs day of the week")
plt.xlabel("Day of the week")
plt.ylabel("Frequence of purchase")


# In[193]:


# seems all day of the week are same with respective of purchasing the FD
# lets see in which month, customer purchased FD more
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.month, df.y).plot(kind="bar")
plt.title("Purchase of frequency vs day of the week")
plt.xlabel("Day of the week")
plt.ylabel("Frequence of purchase")


# In[83]:


#seems in middle of year from april till Aug, sale was more
#lets see the customer age distribution in given data
df.age.hist()
plt.title("customer age distribution in given data")
plt.xlabel("age")
plt.ylabel("count")


# df.head()

# In[84]:


### we can see above, majority of customer have age between 30 years to 40 years
#we can see many variable have text data in that, so it means they are categorical variables. 
# Data models works best in numerical data, so we would need to convert those categorical variables into numeric data
# we will create dummary variable for the same.
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for variable in cat_vars:
    cat_list='var'+'_'+variable
    cat_list = pd.get_dummies(df[variable], prefix=variable)
    df1=df.join(cat_list)
    df=df1

df.head()


# In[94]:


#now delete original columns as we already got these dummey variables with respective data
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
df_final=df[to_keep]
df_final.columns.values


# In[100]:


print(len(to_keep))


# In[114]:


#we can see number of columns reduced to 62.
# now lets try to balance the data set, as we know ~88% costomer didnt buy FD but ~12% customer bought FD
#we will try to use SMOTE technique
from imblearn.over_sampling import SMOTE
X = df_final.loc[:, df_final.columns != 'y']
y = df_final.loc[:, df_final.columns == 'y']
os = SMOTE(random_state=0)


# In[118]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[122]:


columns = X_train.columns


# In[145]:


#now we are over sampling only training data set
os_data_X,os_data_y=os.fit_resample(X_train, y_train)


# In[131]:


os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])


# In[144]:


print("length of oversample data", len(os_data_X))
print("no of no subscription in oversampled data", len(os_data_y[os_data_y["y"]==0]))
print("no of subscription in oversampled data", len(os_data_y[os_data_y["y"]==1]))
print("% of subscription in oversampled data", (len(os_data_y[os_data_y["y"]==1]))/len(os_data_X))
print("% of no subscription in oversampled data", (len(os_data_y[os_data_y["y"]==0]))/len(os_data_X))


# In[ ]:


#as per above data, our over sample data set is now balanced
# further we will try to select best features which will be use ful for our model prediction
df_final_vars = df_final.columns.values.tolist()
y=['y']
X=[i for i in df_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']


# In[194]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[209]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train) 
pipe.score(X_test, y_test)

#X_train = preprocessing.StandardScaler().fit(X_train)


# In[ ]:





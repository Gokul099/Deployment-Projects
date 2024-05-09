import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('HeartDisease.csv')
df.head()

"""# Basic Check"""

df.head()

df.shape

df.info()

df.isnull().sum()

"""## In this data no null values and no categorical values."""

df.describe()

"""# EDA"""

plt.figure(figsize = (12,12))
sns.countplot(data = df,x = 'age', palette = sns.color_palette('dark'))
plt.xlabel('Age',fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

"""### 54 - 60 age people is more.
### 29, 74 - 77 age people is less compare than other age.
"""

plt.figure(figsize = (12,10))
sns.countplot(data = df, x = 'gender', palette = sns.color_palette('bright'))
plt.xlabel("Gender",fontsize = 20)
plt.show()

"""### Male is more than female.
### Female is the half count of male.
"""

plt.figure(figsize = (14,12))
sns.countplot(data = df,x = 'chest_pain', palette = sns.color_palette('magma'))
plt.xlabel("Chest Pain", fontsize = 20)
plt.show()

"""### More patient not experienced chest pain.
### But some patient experienced 2nd pain.
### 3rd is a last pain after they death.
"""

plt.figure(figsize = (13,10))
sns.countplot(data = df, x= 'rest_bps')
plt.xlabel('Rest BB', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

"""### The BB rate is 120 is normal. In this 120 rate patient is more.
### But above 120 is high BB rate, in this more patients have 130 - 150.
### Less patient have low BB.
"""

plt.figure(figsize = (12,12))
sns.histplot(data = df,x = 'cholestrol')
plt.xlabel('Cholestrol', fontsize = 20)
plt.show()

"""### 200 - 240 is boderline.
### Above 240 is high,
### Below 200  is normaL
### In this above 240mg cholestrol patients is more, they have in risk level.
"""

plt.figure(figsize = (13,12))
sns.countplot(data = df, x = 'fasting_blood_sugar', palette = sns.color_palette('cividis'))
plt.xlabel('Fasting Blood Sugar',fontsize = 20)
plt.show()

"""### patients have more no sugar in fasting or before eating food."""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'rest_ecg', palette = sns.color_palette('Accent'))
plt.xlabel('Rest ECG',fontsize  = 20)
plt.show()

"""### more patients have normal ECG 0 and 1."""

plt.figure(figsize = (12,13))
sns.histplot(data = df, x = 'thalach')
plt.xlabel('Thalach', fontsize = 20)
plt.show()

"""### In thalach more in 140 - 170.
### less in 75 - 80.
"""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'exer_angina', palette = sns.color_palette('Accent'))
plt.xlabel('Exercise-induced Angina', fontsize = 20)
plt.show()

"""### In this NO is majority"""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'old_peak', palette = sns.color_palette('Set1'))
plt.xlabel('Old Peak', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

"""### 0.0 is more number."""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'slope', palette = sns.color_palette('Set2'))
plt.xlabel('Slope', fontsize = 20)
plt.show()

"""### 1 and 2 is more. they both are equal."""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'ca', palette = sns.color_palette('Set3'))
plt.xlabel('major vessels', fontsize = 20)
plt.show()

"""### 0 is more."""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'thalassemia', palette = sns.color_palette('Set2'))
plt.xlabel('Thalassemia', fontsize = 20)
plt.show()

"""### 2 is more, and 0 is least count."""

plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'target', palette = sns.color_palette('dark'))
plt.xlabel('Target', fontsize = 20)
plt.show()

"""### In this, more postive patients are there.

# Problem Solving:
"""

df.head()

df['target'].replace({0 : 'no', 1 : 'yes'}, inplace = True)

plt.figure(figsize = (12,13))
sns.histplot(data = df, x = 'age', hue = 'target', palette = sns.color_palette('dark'))
plt.show()

"""### In age 40 - 55 age patient more affected by heart problems."""

pal = ('red','green')
plt.figure(figsize = (10,11))
sns.scatterplot(data = df, x = 'age', y  = 'cholestrol',hue = 'target', palette = pal )
plt.show()

"""### more cholestrol is conformly affeced heart attack.
### 100 - 300 range cholestrol and in age 30 - 55 more affected.
"""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'gender', hue = 'target', palette = pal)
plt.show()

"""### In female positive is more and negative is less.
### And same time, In male postive is less and negative is more.
"""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'chest_pain', hue = 'target', palette = pal)
plt.show()

"""### Chest pain 2 is more dangours many patients postive in 2.
### And 0 and 1 is a equal range.
"""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.histplot(data = df, x = 'rest_bps', hue = 'target', palette = pal)
plt.show()

"""### In rest bb 120 - 140 and 100 - 110 range patient is more affected bu heart disease.
### 160 - 200 range patient is no affected heart disease.
"""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'fasting_blood_sugar', hue = 'target', palette = pal)
plt.show()

"""### In fasting blood sugar 0 is have more postive heart disease."""

plt.figure(figsize = (12,13))
sns.histplot(data = df, x = 'thalach', hue = 'target', palette = sns.color_palette('dark'))
plt.show()

"""### 150 - 200 range is more affected by heart disease."""

df.head(2)

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'exer_angina', hue = 'target', palette = pal)
plt.show()

"""### In 0 is have more postive heart disease"""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'old_peak', hue = 'target', palette = pal)
plt.show()

"""### 0.0 - 1.6 is more in postive heart disease"""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'slope', hue = 'target', palette = pal)
plt.show()

"""### range 2 is more postive heart disease."""

pal = ('red','green')
plt.figure(figsize = (12,13))
sns.countplot(data = df, x = 'thalassemia', hue = 'target', palette = pal)
plt.show()

"""### In thalassemia range 2 is more affected in heart disease.

# preprocessing
"""

plt.figure(figsize=(10, 10))
plotnumber = 1
for col in df:
    if plotnumber <= len(col):
        plt.subplot(4, 4, plotnumber)
        sns.boxplot(df[col].dropna(axis=0))
        plt.xlabel(col)
        plt.ylabel('count')
        plotnumber += 1

plt.tight_layout()
plt.show()

q1 = df['rest_bps'].quantile(0.25)
print('low:',q1)
q3 = df['rest_bps'].quantile(0.75)
print('high:',q3)

iqr = q3-q1
iqr

low = q1-1.5*iqr
print('low:',low)
high = q3+1.5*iqr
print('high:',high)

df.loc[df['rest_bps']>high]

df.loc[df['rest_bps']<low]

df.loc[df['rest_bps']>high, 'rest_bps'] = df['rest_bps'].median()

df.loc[df['rest_bps']>high]

q1 = df['cholestrol'].quantile(0.25)
print('low:',q1)
q3 = df['cholestrol'].quantile(0.75)
print('high:',q3)

iqr = q3 - q1
iqr

low = q1-1.5*iqr
print('low:',low)
high = q3+1.5*iqr
print('high:',high)

df.loc[df['cholestrol']>high]

df.loc[df['cholestrol']<low]

df.loc[df['cholestrol']>high,'cholestrol'] = df['cholestrol'].median()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace = True)

df['target'].replace({'no' : 0, 'yes' : 1}, inplace = True)

a = df.corr()
print(a)

"""# Spliting"""

x = df.drop('target',axis = 1)
y = df['target']

x

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

x_train.shape

y_train.shape

"""# Model Creation

# Decision Tree
"""

from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier()
dc.fit(x_train,y_train)

#Training
x1 = dc.predict(x_train)
x1

# Testing
y1 = dc.predict(x_test)
y1

print(f'Training score:{accuracy_score(y_train,x1)}')
print(f'Testing score:{accuracy_score(y_test,y1)}')

print(f'TR:{classification_report(y_train,x1)}')
print(f'TS:{classification_report(y_test,y1)}')

"""# Random Forest"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

x2 = rf.predict(x_train)
x2

y2 = rf.predict(x_test)
y2

print(f'TR:{classification_report(y_train,x2)}')
print(f'TS:{classification_report(y_test,y2)}')

"""## Hyperparameter Tunning"""

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rc = RandomForestClassifier()
zx = GridSearchCV(rc,param_grid ,n_jobs = -1,cv = 5)
zx.fit(x_train,y_train)

w = zx.best_params_
print(w)

ran = RandomForestClassifier(bootstrap = True,max_depth  = 20, max_features = 'log2',min_samples_leaf = 4, min_samples_split = 5,n_estimators = 50)
ran.fit(x_train,y_train)

# TRaining
x3 = ran.predict(x_train)
x3

# Testing
y3 = ran.predict(x_test)
y3

print(f'TR:{classification_report(y_train,x3)}')
print(f'TS:{classification_report(y_test,y3)}')

import joblib

file_name = 'heart diseses' 
joblib.dump(ran,'heart diseses')
app = joblib.load('heart diseses')
arr = [[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]
d = app.predict(arr)
print(d)

import pickle
Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(ran, file)

with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

Pickled_Model


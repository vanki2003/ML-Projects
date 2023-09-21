import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('datasets\skyserver_sdss_dr16_100k\skyserver_sdss_dr16_100k.csv', skiprows=1)
data.head()

# Drop columns with technical IDs and dates
data = data.drop(columns=['objid','run','rerun','camcol','field','plate','fiberid','mjd'])
data.head()

data.describe()

# Remove outliers from psfMag_? columns
for c in ['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']: 
    lower = data[c].quantile(0.01)
    upper = data[c].quantile(0.99)

    data = data[(data[c] < upper) & (data[c] > lower)]

data.describe()

# number of records per class is equal
records = len(data[data['class'] == 'QSO'])
print(records)

data = data.groupby('class').sample(n=records)
data.describe()

#training the model

#declare variables for training
x_data = data['class']
y_data = data.drop(columns=['class'])

#transforming class names to numerical values
le = preprocessing.LabelEncoder()
x_data=le.fit_transform(x_data)

#standardizing the feature values
ss = preprocessing.StandardScaler()
scale = ss.fit_transform(data.drop(columns='class'))
print(scale)

#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

#creating the model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(x_train, y_train)

#evauling the model
print(model.score(x_test, y_test))
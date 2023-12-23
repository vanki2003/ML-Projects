import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('datasets\winequality-red.csv', delimiter=';')
data.head()

for c in data:
    if  c != 'quality':

        fig, axs = plt.subplots(nrows = 3, sharex =True, figsize =(8,6))

#plotting histogram
        data[c].plot.hist(ax=axs[0], bins=30, grid = True)
        axs[0].set_title('Alcohol')

#plotting KDE
        data[c].groupby(by=[data['quality']]).plot.density(ax=axs[1], grid = True, legend=True)
        axs[1].legend(title='Wine Quality', loc=1, ncols=2, frameon=True, facecolor='white')

#plotting boxplot
        data[c].plot.box(ax=axs[2], vert = False, grid = True, label='')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_data=data.drop(columns=['quality'])
y_data = data['quality']

#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

#creating and training the model
model = LinearRegression()
model.fit(x_train, y_train)

#evaluating the model
print(model.score(x_test,y_test))

print(model.intercept_)
print(model.coef_)

y_hat=model.predict(x_test)

#plotting results
plt.hist([y_test, y_hat], bins=10, color=['blue', 'green'], label=['True', 'Predicted'], alpha=0.7)
plt.xlabel("Wine Quality")
plt.ylabel("Frequency")
plt.legend()
plt.show()
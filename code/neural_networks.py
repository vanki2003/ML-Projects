import numpy as np
import pandas as pd

data = pd.read_excel('datasets\Dry_Bean_Dataset.xlsx',engine='openpyxl')
data.head()

x_data = data.drop(columns='Class')
y_data = data['Class']

classes = y_data.to_numpy()
print(classes)
print(classes.shape)

unique_classes=np.unique(classes)
print(unique_classes)

new_array = np.zeros((len(classes), len(unique_classes)))
new_array.shape
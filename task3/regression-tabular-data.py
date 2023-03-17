import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train_file = 'internship_train.csv'
test_file = 'internship_hidden_test.csv'

df_data_train = pd.read_csv(train_file, low_memory=False)
df_data_test = pd.read_csv(test_file)

X = np.array(df_data_train.drop('target', axis=1))
y = np.array(df_data_train['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=34)

model_rf = RandomForestRegressor(max_depth=20, n_estimators=20, n_jobs=1)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

y_pred_result = model_rf.predict(df_data_test.values)
y_pred_result = pd.DataFrame(y_pred_result).to_csv('predictions.csv', index=None)

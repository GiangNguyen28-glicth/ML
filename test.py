import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
data = pd.read_csv('./train-data.csv',index_col="CarId")
data.shape
x, y = train_test_split(data, test_size=0.2, random_state=42)

y_train=x["Price"].copy()
y_train

x_train = x.drop(columns = "Price")
x_train

y_valid = y["Price"].copy()
y_valid

x_valid = y.drop(columns = "Price")
x_valid
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values  

features_numbers = ["Year","Kilometers_Driven","Seats"]
features_cat =["Name","Location","Fuel_Type","Transmission","Owner_Type","Mileage","Engine","Power","New_Price"]

cat_pipeline = Pipeline([
    ('selector', ColumnSelector(features_cat)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)),
    ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))])

num_pipeline = Pipeline([
    ('selector', ColumnSelector(features_numbers)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))])   

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)])


processed_train_set_val = full_pipeline.fit_transform(x_train)
processed_train_set_val.shape
processed_test_set_val = full_pipeline.transform(x_valid)
processed_test_set_val.shape
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(features_numbers)))

from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(random_state=1)
dt_model.fit(processed_train_set_val,y_train)

print("\nPredictions: ", dt_model.predict(processed_test_set_val[0:9]))
print("Labels:      ", list(y_valid[0:9]))
# def r2score_and_rmse(model, train_data, labels): 
#     r2score = model.score(train_data, labels)
#     from sklearn.metrics import mean_squared_error
#     prediction = model.predict(train_data)
#     mse = mean_squared_error(labels, prediction)
#     rmse = np.sqrt(mse)
#     return r2score, rmse      
# r2score, rmse = r2score_and_rmse(dt_model, processed_train_set_val, y_train)
# print('\nR2 score (on training data, best=1):', r2score)
# print("Root Mean Square Error: ", rmse.round(decimals=1))


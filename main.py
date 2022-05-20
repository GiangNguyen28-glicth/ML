import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
data = pd.read_csv('./train-data.csv',index_col="CarId")
data.head()
for col in data.columns:
    missing_data=data[col].isna().sum()
    missing_persent=missing_data/len(data)*100
    print(f"Column: {col} has {missing_persent}%")
features = ["Name","Year","Kilometers_Driven","Seats"]
x=data[features].values
y=data.iloc[:,-1].values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan,strategy="mean")
impute.fit(x[:,1:4])
x[:,1:4]=impute.transform(x[:,1:4])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
x=ct.fit_transform(x)
x
x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(random_state=1)
dt_model.fit(x_train,y_train)
y_preds = dt_model.predict(x_valid)
pd.DataFrame({'y':y_valid,'y_preds':y_preds})





impute = SimpleImputer(missing_values=np.nan,strategy="mean")
impute.fit(x[:,10:11])
x[:,10:11]=impute.transform(x[:,10:11])
x=data[features]
y=data["Price"]
x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(random_state=1)
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan,strategy="mean")
impute.fit(x[:,11:12])
x[:,11:12]=impute.transform(x[:,11:12])
dt_model.fit(x_train,y_train)
y_preds = dt_model.predict(x_valid.head())
pd.DataFrame({'y':y_valid.head(),'y_preds':y_preds})
features = ["Year","Kilometers_Driven","Seats"]
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values 
num_pipeline = Pipeline([
    ('selector', ColumnSelector(features)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True))])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline)]) 

y=data["Price"].values
y
processed_train_set_val = full_pipeline.fit_transform(y)
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)



import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.model_selection import GridSearchCV
import seaborn as sns
data = pd.read_csv('./train-data.csv',index_col="CarId")
data.info()
categories_cols = data.select_dtypes(include=['object']).columns

#==================Xem phần trăm missing data trong mỗi cột ===========================
for col in data.columns:
    missing_data=data[col].isna().sum()
    missing_persent=missing_data/len(data)*100
    print(f"Column: {col} has {missing_persent}%")
#==============================Tách dữ liệu ra thành tập train và tập test========================
x, y = train_test_split(data, test_size=0.2, random_state=42)
y_train=x["Price"].copy()
x_train = x.drop(columns = "Price")
y_valid = y["Price"].copy()
x_valid = y.drop(columns = "Price")

#============================Xử lý missing data ========================================
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values  

#====================Liệt kê các Numerical Features 
features_numbers = list(data.select_dtypes(include=[np.number]))
#====================Liệt kê các Categorical Features
features_cat = list(data.select_dtypes(exclude=[np.number]))
features_numbers = features_numbers[:3]

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


processed_train_set_val = full_pipeline.fit_transform(x_train)  # x_train
processed_test_set_val = full_pipeline.transform(x_valid)     # x_test

print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(features_numbers)))
#==================================Name====================================
plt.figure(figsize=(5,5))
data["Name"].value_counts().plot(kind='bar')
plt.show()
#==================================Price===================================
plt.figure(figsize=(5,5))
sns.distplot(data["Price"],kde=True)
plt.show()
#=================================Draw=====================================
for i in categories_cols:
    if i!='Name':
        sns.boxplot(x=i,y='Price',data=data,palette='Pastel2')
        plt.show()
#================================Price & Year=============================
year,price = data["Year"],data["Price"]
plt.scatter(year,price)
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()
#===============================DecisionTreeRegressor===============================
from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(random_state=1)
dt_model.fit(processed_train_set_val[0:9],y_train[0:9])
dt_model.get_params()
print("\nPredictions: ", dt_model.predict(processed_test_set_val[0:9]))
print("Labels:      ", list(y_valid[0:9]))

def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
accuracy_train_p, rmse_train = r2score_and_rmse(dt_model, processed_train_set_val, y_train)
print("Độ chính xác trên tập huấn luyện :",accuracy_train_p)
print("RMSE :",rmse_train)
accuracy_test_p, rmse_test = r2score_and_rmse(dt_model, processed_test_set_val, y_valid)
print("Độ chính xác trên tập test :",accuracy_test_p)
print("RMSE :",rmse_test)

print('\n____________ K-fold cross validation ____________')
cv = KFold(n_splits=5,shuffle=True,random_state=37)
dt_model=DecisionTreeRegressor(random_state=1)
#Train
nmse_scores_train = cross_val_score(dt_model, processed_train_set_val, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores_train = np.sqrt(-nmse_scores_train)
print("DecisionTreeRegressor rmse: ", rmse_scores_train.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores_train.round(decimals=1)),'\n')

#Test
nmse_scores_test = cross_val_score(dt_model, processed_test_set_val, y_valid, cv=cv, scoring='neg_mean_squared_error')
rmse_scores_test = np.sqrt(-nmse_scores_test)
print("DecisionTreeRegressor rmse: ", rmse_scores_test.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores_test.round(decimals=1)),'\n')

model_name = "DecisionTreeRegressor"
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params)   

# param_grid = {
#             "max_depth" : np.arange(10,100) }
param_grid={'max_depth': [5],
 'max_features': 'auto',
 'max_leaf_nodes': [40],
 'min_samples_leaf': [2],
 'min_weight_fraction_leaf': [0.1],
 'splitter': ['random']}
cv = KFold(n_splits=2,shuffle=True,random_state=37)          
grid_search = GridSearchCV(dt_model,param_grid=param_grid,scoring='neg_mean_squared_error',cv=3,verbose=3)
grid_search.fit(processed_train_set_val[0:9], y_train[0:9])
print_search_result(grid_search,model_name)
print('Best hyperparameter combination: ',grid_search.best_params_)
print('Best rmse: ', np.sqrt(-grid_search.best_score_)) 
#%%
import joblib
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.model_selection import GridSearchCV
import seaborn as sns
data = pd.read_csv('./train-data.csv')
#%%
#==================Xem % missing data trong mỗi cột ===========================
for col in data.columns:
    missing_data=data[col].isna().sum()
    missing_persent=missing_data/len(data)*100
    print(f"Column: {col} has {missing_persent}%")

#==================Loại bỏ các cột không cần thiết=============================
data = data.drop(columns=["CarId","Name","New_Price"])
data.info()
categories_cols = data.select_dtypes(include=['object']).columns

#================== Ép kiểu dữ liệu ===============================
data['Engine']=data["Engine"].str.replace(' CC','')
data['Power']=data["Power"].str.replace(' bhp','')
data.head()
data['Engine']=data["Engine"].astype(np.float).astype("float64")
data['Power']=data['Power'].astype(np.float).astype("float64")
data.info()
#================== Tách dữ liệu ra thành tập train và tập test========================
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

#====================Liệt kê các Numerical Features ======================================
features_numbers = list(data.select_dtypes(include=[np.number]))
features_numbers = features_numbers[:4]
features_numbers
#====================Liệt kê các Categorical Features
features_cat = list(data.select_dtypes(exclude=[np.number]))
features_cat
#====================Xử lý Missing data và StandardScaler======================================
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
#%%
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(features_numbers)))
#==================================Price===================================
plt.figure(figsize=(5,5))
sns.distplot(data["Price"],kde=True)
plt.show()
#==================================Fuel_Type===================================
plt.figure(figsize=(7,7))
data['Fuel_Type'].value_counts().plot(kind='bar')
plt.show()
#==================================Owner_Type===================================
plt.figure(figsize=(7,7))
data['Owner_Type'].value_counts().plot(kind='bar')
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
#================================Nguyen====================================
# sns.set_theme()
sns.histplot(data=data["Price"])
plt.show()

sns.barplot(data=data,x="Owner_Type",y="Price")
plt.xticks(rotation='vertical')
plt.show()

sns.barplot(data=data,x="Transmission",y="Price")
plt.xticks(rotation='vertical')
plt.show()

sns.barplot(data=data,x="Fuel_Type",y="Price")
plt.xticks(rotation='vertical')
plt.show()

sns.barplot(data=data,x="Price",y="Location")
plt.show()
#%%
#=============================K-fold cross validation=============================================================================
print('\n____________ K-fold cross validation ____________')
cv = KFold(n_splits=5,shuffle=True,random_state=37)
#===============================Tính Score và RMSE================================================================================
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse
#==================================Fine-tune models================================================================================
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

#%%
#===============================Train GradientBoostingRegressor===========================
print("\n____________ Train GradientBoostingRegressor ____________")
grabt_model = GradientBoostingRegressor(random_state = 42)
grabt_model.fit(processed_train_set_val,y_train)
print("\nPredictions: ", grabt_model.predict(processed_test_set_val[0:9]))
print("Labels:      ", list(y_valid[0:9]))
#===============================Tính Score và RMSE trên GradientBoostingRegressor======================
#%%
print("\n____________ Tính Score và RMSE trên GradientBoostingRegressor ____________")
accuracy_train_grabt, rmse_train_grabt = r2score_and_rmse(grabt_model, processed_train_set_val, y_train)
print("Độ chính xác trên tập huấn luyện :",accuracy_train_grabt)
print("RMSE :",rmse_train_grabt)
accuracy_test_p, rmse_test = r2score_and_rmse(grabt_model, processed_test_set_val, y_valid)
print("Độ chính xác trên tập test :",accuracy_test_p)
print("RMSE :",rmse_test)
#=============================K-fold cross validion trên GradientBoostingRegressor======================
#%%
print("\n____________ K-fold cross validion trên GradientBoostingRegressor trên tập Train ____________")
#Train
nmse_scores_train_kfoldgrd = cross_val_score(grabt_model, processed_train_set_val, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores_train_kfoldgrd = np.sqrt(-nmse_scores_train_kfoldgrd)
print("GradientBoostingRegressor rmse: ", rmse_scores_train_kfoldgrd.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores_train_kfoldgrd.round(decimals=1)),'\n')
#%%
#Test
print("\n____________ K-fold cross validion trên GradientBoostingRegressor trên tập Test ____________")
nmse_scores_test = cross_val_score(grabt_model, processed_test_set_val, y_valid, cv=cv, scoring='neg_mean_squared_error')
rmse_scores_test = np.sqrt(-nmse_scores_test)
print("GradientBoostingRegressor rmse: ", rmse_scores_test.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores_test.round(decimals=1)),'\n')

#==============================Fine-tune models GradientBoostingRegressor=======================
#%%
print("\n____________ Fine-tune models GradientBoostingRegressor ____________")
print("\n____________ RandomizedSearchCV ____________")
loss = ['ls','lad','huber']
n_estimators = [100,500, 900, 1100,1500]
max_depth = [2,3,5,10,15]
min_samples_leaf = [1,2,4,6,8]
min_samples_split = [2, 4, 6, 10]
max_features = ['auto', 'sqrt', 'log2', None]
hyperparameter_grid = {
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}
random_cv = RandomizedSearchCV(estimator=grabt_model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

load_randomcv=False
if load_randomcv: 
    random_cv.fit(processed_train_set_val, y_train)
    random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)
    random_results.head(10)
    random_cv.best_estimator_
    joblib.dump(random_cv,'models/GradientBoostingRegressor_RandomizedSearchCV.pkl') 
else:
    random_cv = joblib.load('models/GradientBoostingRegressor_RandomizedSearchCV.pkl')


#%%
print("\n____________ GradientBoostingRegressor ____________")
grabt_model = GradientBoostingRegressor(max_depth = 5,
                                  min_samples_split = 10,
                                  max_features = None,
                                  random_state = 42)
#%%
print("\n____________ GridSearchCV ____________")
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
grid_search = GridSearchCV(estimator = grabt_model, 
                           param_grid=trees_grid, 
                           cv = 4, 
                           scoring = 'neg_mean_absolute_error', 
                           verbose = 1,
                           n_jobs = -1, 
                           return_train_score = True)
grid_search.fit(processed_train_set_val, y_train)
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(5,5))
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend()
plt.title('Performance vs Number of Trees')
results.sort_values('mean_test_score', ascending = False).head(5)
# %%
#==============================Final Model GradientBoostingRegressor=======================
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
print("\n____________ Final Model GradientBoostingRegressor ____________")
default_model_grabt = GradientBoostingRegressor(random_state = 42)
final_model_grabt = grid_search.best_estimator_

default_model_grabt.fit(processed_train_set_val, y_train)
final_model_grabt.fit(processed_train_set_val, y_train)

default_pred = default_model_grabt.predict(processed_test_set_val)
final_pred = final_model_grabt.predict(processed_test_set_val)


print(final_pred[0:9])
print(final_model_grabt.score(processed_test_set_val,y_valid))
print(default_model_grabt.score(processed_test_set_val,y_valid))
print(list(y_valid[0:9]))
print('Default model performance on the test set: MAE = %0.4f.' % mae(y_valid, default_pred))
print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_valid, final_pred))
accuracy_train_grabt_t, rmse_train_grabt_t = r2score_and_rmse(final_model_grabt, processed_train_set_val, y_train)
print("GradientBoostingRegressor Default rmse: ", accuracy_train_grabt_t.round(decimals=1))
print("Avg. rmse: ", mean( rmse_train_grabt_t.round(decimals=1)),'\n')

# accuracy_test_grabt_f, rmse_test_grabt_f = r2score_and_rmse(final_model_grabt,processed_test_set_val, y_valid)
# print("GradientBoostingRegressor Final rmse: ", accuracy_test_grabt_f.round(decimals=1))
# print("Avg. rmse: ", mean(rmse_test_grabt_f),'\n')

sns.kdeplot(final_pred, label = 'Predictions')
sns.kdeplot(y_valid, label = 'Values')
plt.xlabel('Energy Star Score'); 
plt.ylabel('Density')
plt.title('Test Values and Predictions')

print('Default model performance on the test set: MAE = %0.4f.' % mae(y_valid, default_pred))
print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_valid, final_pred))

#==============================RandomForestRegressor=======================
#%% 5.3 Try RandomForestRegressor model
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model_random = RandomForestRegressor(n_estimators = 5) # n_estimators: no. of trees
model_random.fit(processed_train_set_val, y_train)
# Compute R2 score and root mean squared error
#%%
print('\n____________ RandomForestRegressor ____________')
r2score_model_random, rmse_model_random = r2score_and_rmse(model_random, processed_train_set_val, y_train)
print('\nR2 score (on training data, best=1):',r2score_model_random)
print("RMSE trên tập train: ", rmse_model_random.round(decimals=1))     
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model_random.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(y_train[0:9]))
#%% 5.5 Evaluate with K-fold cross validation 
print('\n____________ K-fold --> rmse ____________')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator: just a try to persist data splits (hopefully)
model_name = "RandomForestRegressor" 
model_random = RandomForestRegressor(n_estimators = 5)
nmse_scores_train_kfoldrandom = cross_val_score(model_random, processed_train_set_val, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores_train_kfoldrandom = np.sqrt(-nmse_scores_train_kfoldrandom)
print("RandomForestRegressor rmse: ", rmse_scores_train_kfoldrandom.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores_train_kfoldrandom.round(decimals=1)),'\n')

# %% Fine-turn tìm ra hyperparameter định hình nên model ( phải đưa dô trc khi training )
from sklearn.model_selection import GridSearchCV
cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator
model_random = RandomForestRegressor()
param_grid = [
    # try 15 (3x4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
    {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
    # then try 12 (4x3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]
    # Train across 5 folds, hence a total of (15+12)*5=135 rounds of training 
grid_search = GridSearchCV(model_random, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
refit=True) # refit=True: after finding best hyperparam, it fit() the model with whole data (hope to get better result)
grid_search.fit(processed_train_set_val, y_train)
print_search_result(grid_search, model_name = "RandomForestRegressor")   
#%%
#==============================Final Model GradientBoostingRegressor=======================
final_model_random = grid_search.best_estimator_
final_model_random.fit(processed_train_set_val,y_train)
final_pred_random = final_model_random.predict(processed_test_set_val)
print(final_pred_random[0:9])
print(list(y_valid[0:9]))

accuracy_train_final_random, rmse_train_final_random = r2score_and_rmse(final_model_random, processed_train_set_val, y_train)
print("Score: ", accuracy_train_final_random.round(decimals=1))
print("RMSE",rmse_train_final_random,'\n')
# %%
#==============================Feature Importances=======================
best_model = grid_search.best_estimator_
feature_importances = best_model.feature_importances_
feature_names = x_train.columns.tolist()
for name in features_cat:
    feature_names.remove(name)
print('\nFeatures and importance score: ')
values_importance = []
feature_names_new=[]
for importance,name in sorted(zip(feature_importances,feature_names),reverse=True):
    values_importance.append(importance)
    feature_names_new.append(name)
    print((importance,name))
feature_results = pd.DataFrame({'feature': feature_names_new, 
                                'importance':  values_importance})
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)
plt.figure(figsize=(5,5))
feature_results.loc[:9, :].plot(x = 'feature', y = 'importance', 
                                 edgecolor = 'k',
                                 kind='barh', color = 'blue')
plt.xlabel('Relative Importance', size = 20)
plt.ylabel('')
plt.title('Feature Importances from Random Forest', size = 30)
    # Print features and importance score  (ONLY on rand forest)
    
# %%
model_comparison = pd.DataFrame({'model':['Gradient Boosted','RandomForestRegressor'],
                                'RMSE':[rmse_train_grabt,rmse_model_random]})
model_comparison.sort_values('RMSE',ascending = False).plot(x = 'model',
                                                           y = 'RMSE',
                                                           kind = 'barh',
                                                           color = 'red', 
                                                           edgecolor = 'black')
plt.ylabel('')
plt.yticks(size = 14)
plt.xlabel('RMSE before K-Fold')
plt.xticks(size = 14)
plt.title('RMSE', size = 20)
plt.show()

#%%
avg_rmse_kfoldgrd=mean(rmse_scores_train_kfoldgrd.round(decimals=1))
avg_rmse_kfoldrandom=mean(rmse_scores_train_kfoldrandom.round(decimals=1))
model_comparison = pd.DataFrame({'model':['Gradient Boosted','RandomForestRegressor'],
                                'RMSE':[avg_rmse_kfoldgrd,avg_rmse_kfoldrandom]})
model_comparison.sort_values('RMSE',ascending = False).plot(x = 'model',
                                                           y = 'RMSE',
                                                           kind = 'barh',
                                                           color = 'red', 
                                                           edgecolor = 'black')
plt.ylabel('')
plt.yticks(size = 14)
plt.xlabel('RMSE after K-Fold')
plt.xticks(size = 14)
plt.title('RMSE', size = 20)
plt.show()
# %%
most_important_features = feature_results['feature'][:5]
single_tree = final_model_grabt.estimators_[105][0]
processed_train_set_val_most_importance = x_train[most_important_features]
processed_train_set_val_most_importance.isna().sum()
processed_train_set_val_most_importance.fillna(0,inplace=True)
num_pipeline = Pipeline([
    ('selector', ColumnSelector(processed_train_set_val_most_importance)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))])   

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline)])
ok = full_pipeline.fit_transform(processed_train_set_val_most_importance)
processed_train_set_val_most_importance.shape
final_model_grabt.fit(processed_train_set_val_most_importance,y_train)
tree.export_graphviz(single_tree, 
                     out_file = './images/tree.dot',
                     rounded = True, 
                     feature_names = most_important_features,
                     filled = True)
nmse_scores_test = cross_val_score(final_model_grabt, processed_train_set_val, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores_test = np.sqrt(-nmse_scores_test)
print("GradientBoostingRegressor rmse: ", rmse_scores_test.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores_test.round(decimals=1)),'\n')

# %%

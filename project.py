import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv(r"data_set.data",names = ['sym','loss','make','fuel_t','asp','n_doors','body','wheels','engine_l','wheel_b','length','width','height','c_height','engine_t','n_cylinders','engine_s','fuel_s','bore','stroke','c_ratio','horsepower','p_rpm','c_mpg','h_mpg','price'], na_values = ['?'])
features = df.drop('sym', 1)
pred = df['sym']
numeric_cols = ['loss','stroke','horsepower','p_rpm','price','bore']
string_cols = ['make','fuel_t','asp','n_doors','body','wheels','engine_l','engine_t','n_cylinders','fuel_s']
#dummies_cols = ['d_make','d_fuel_t','d_asp','d_n_doors','d_body','d_wheels','d_engine_l','d_engine_t','d_n_cylinders','d_fuel_s']
for col in numeric_cols:
    features[col] = features[col].fillna(features[col].mean())
n_mode = features['n_doors'].mode()
features[['n_doors']] = features[['n_doors']].replace(np.nan,n_mode[0])
# for column in features:
#      print(column)
#      print(features[column].value_counts(dropna=False))
for col in string_cols:
    if col == 'n_doors' :
        features = features.join(pd.get_dummies(features[col],prefix='nd'))
    elif col == 'n_cylinders' :
        features = features.join(pd.get_dummies(features[col],prefix='nc'))
    else:
        features = features.join(pd.get_dummies(features[col]))
features = features.drop(string_cols,1)
features_train, features_test, pred_train, pred_test = train_test_split(features, pred, test_size=0.25, random_state = 0)
scale_features = StandardScaler()
features_train = scale_features.fit_transform(features_train)
features_test = scale_features.transform(features_test)
clf1= LogisticRegression(random_state=0,solver='saga',max_iter=20000)
clf2= LogisticRegression(penalty = 'l1', random_state=0,solver='saga',max_iter=20000)
clf1.fit(features_train,pred_train)
clf2.fit(features_train,pred_train)
y_pred = clf1.predict(features_test) 
y_pred = clf2.predict(features_test) 
print(clf1.score(features_train,pred_train))
print(clf2.score(features_train,pred_train))
data = [93.0,106.7,187.5,70.3,54.9,3495,183,3.58,3.64,21.5,123.0,4350.0,22,25,28176.0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
y_pred1 = clf2.predict(np.array(data).reshape(1,-1))
print(y_pred1)
y_pred2= clf2.predict(np.array(data).reshape(1,-1))
print(y_pred2)
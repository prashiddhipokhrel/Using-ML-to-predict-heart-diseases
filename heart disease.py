import pandas as pd
import numpy as np 
import os 
from sklearn import metrics
import seaborn as sbs 
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler as SS 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
 
 
## load the file ##
os.chdir('C:/Users/Swift/OneDrive/Desktop/projects')
df = pd.read_csv("heart.csv")
 
## check tha data file ##
df.columns
df.shape
df.head()
 
## check if there are any na or missing values in the data set $$ 
df.isnull().any()
df.isna().any()
 
## check if the data set has duplicate columns ## 
df.columns.duplicated().sum()
 
## checking the first column of the dataset ## 
df['age'].value_counts().sum()
 
 
## generating a historgram of the healthiness of datasets ## 
axx = df['target'] 
sbs.set_style('darkgrid')
ax = sbs.countplot(axx , palette = 'magma')
pyplot.xlabel('Target')
pyplot.title('Frequency of Target')
pyplot.show()
 
## general Summary of the DataSet ##
summary = df.describe().T 
summary = summary.round(4)
summary 
 
### correlation matrix ### 
fig, ax = pyplot.subplots(figsize = (20,10))
sbs.heatmap(df.corr(), annot=True, fmt='.1g', cmap="viridis",
           cbar=False, linewidths=0.7, linecolor='black');
 
## checking no of healthy and sick patients ## 
healthy = df[(df['target'] ==0) ].count()[1]
sick = df[(df['target'] == 1)].count()[1]
 
## splitting the data set into x and y for the analysis ## 
y = df.iloc[:, 13:14 ]
x = df.iloc[:, 1:13]
 
## scaling the variables to make the worable ## 
scaler = SS()
x = scaler.fit_transform(x)
x = scaler.transform(x)
 
## seperate into training and testing sets ## 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
 
#### LINEAR REGRESSION #### 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
LinearRegressionScore = lr.score(X_test, y_test)
print("Accuracy obtained by Linear Regression model:",LinearRegressionScore*100) 
## The accuracy score is 51.45 
 
### CROSS VALIDATION ## 
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.50,
                                             random_state = 1)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits = 50)
model = LogisticRegression()
results = cross_val_score(model,x,y,cv = kfold)
print(results) 
results.mean
print("Accuracy obtained by Cross Validation :",results.mean*100)
 
## non-linear methods ## 
## support vector machine ## 
from sklearn import svm 
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = model_selection.train_test_split(x,
                    y, train_size=0.80, test_size=0.20, random_state=101)
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
rbf_pred = rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_pred)
print("Accuracy obtained multiclass SVM :",rbf_accuracy *100)
## the accuracy score is 49.18
 
## KNN ### 
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.50,
                                             random_state = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,
                  y, test_size=0.2, random_state=12345)
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred) 
## the accuracy score is 53.32
 
### MLP ### 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,
                  y, test_size=0.2, random_state=12345)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y,
                                                    random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict(X_test)
score = clf.score(X_test, y_test)
print("Accuracy obtained MLP :",score*100)
## the accuracy score is 86.84 ##

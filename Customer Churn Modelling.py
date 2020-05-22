import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
os.getcwd()
pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',100)
os.chdir('D:\\Graduate Stduy\\Classes 7th Jan - 23rd Mar\\Python\\Group Project'); # eg: D:\\Graduate Stduy\\Assig-3

data = pd.read_csv('Tele_communication_churn.csv')

data.head(10)
data.shape
data.dtypes
DataView = data.head(10)

DataView
target_variable = data['Churn'] 
display(DataView)

X = data.iloc[:,1:]

target = np.array(target_variable)


display(X.head())
data.groupby('Churn')['customerID'].count()

Churn_yes_no = np.array(data.groupby('Churn')['customerID'].count())

Churn_yes = Churn_yes_no[1]
Churn_no = Churn_yes_no[0]



X.describe()


numerical = ['SeniorCitizen', 'tenure', 'MonthlyCharges','TotalCharges']
categorical = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
X[categorical].isnull().any()

df_reduced = X.drop(categorical, axis = 1)
dummy_cat_df = pd.get_dummies(X[categorical], drop_first=True)
df = pd.concat([df_reduced, dummy_cat_df], axis = 1)


df.head()

df.describe()

df.loc[df['Churn'] == 'Yes', 'Churn'] = 1 # approved
df.loc[df['Churn'] == 'No', 'Churn'] = 0 # Not approved
df['Churn'] = df['Churn'].astype('float')
df.dtypes

target_variable = (df['Churn']) 

df_1 = df.drop('Churn', axis=1)
df_1.dtypes


Min_max_scaler = preprocessing.MinMaxScaler()
dp_scaled = Min_max_scaler.fit_transform(df_1)
df_normalize = pd.DataFrame(dp_scaled)
df_normalize
df.rename(index = str, columns= {"1": "" ,})

df_2 = pd.concat([df_normalize, target_variable], axis = 1)
df_r = df_2.copy()


# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data

df_normalize.columns = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','gender_Male','Partner_Yes','Dependents_Yes',
'PhoneService_Yes','MultipleLines_No','MultipleLines_Yes','InternetService_Fiber','InternetService_No',
'OnlineSecurity_No','OnlineSecurity_Yes','OnlineBackup_No','OnlineBackup_Yes','DeviceProtection_No','DeviceProtection_Yes',
'TechSupport_No','TechSupport_Yes','StreamingTV_No','StreamingTV_Yes','StreamingMovies_No','StreamingMovies_Yes',
'Contract_One','Contract_Two','PaperlessBilling_Yes','PaymentMethod_Credit','PaymentMethod_Electronic','PaymentMethod_Mailed']

X = df_r.loc[:, df_r.columns != 'Churn']
y = df_r.loc[:, df_r.columns == 'Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

data_final_vars=df_r.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 10)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

rfe.head()

transform(rfe)


train, test = train_test_split(df_r, test_size = 0.3)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=5.0) 
classifier.fit(train.loc[:, ~(train.columns == 'Churn')], train['Churn'])

pred_train = classifier.predict(train.loc[:, ~(train.columns == 'Churn')])
pred_test = classifier.predict(test.loc[:, ~(test.columns == 'Churn')])
classifier.predict_proba(train.loc[:, ~(train.columns == 'Churn')])
from sklearn.metrics import accuracy_score
ac_train = accuracy_score(pred_train, train['Churn'])
ac_test = accuracy_score(pred_test, test['Churn'])
print("train accuracy : ", ac_train, '\n', "test accuracy : ", ac_test )



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=5.0) 
classifier.fit(train.loc[:, ~(train.columns == 'Class')], train['Class'])


#-------------------------------------------------------------------------------
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )
print(rfe.support_)
print(rfe.ranking_)

cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
X=data_final[cols]
y=data_final['y']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)



# Decision Tree



import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


feature_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','gender_Male','Partner_Yes','Dependents_Yes',
'PhoneService_Yes','MultipleLines_No','MultipleLines_Yes','InternetService_Fiber','InternetService_No',
'OnlineSecurity_No','OnlineSecurity_Yes','OnlineBackup_No','OnlineBackup_Yes','DeviceProtection_No','DeviceProtection_Yes',
'TechSupport_No','TechSupport_Yes','StreamingTV_No','StreamingTV_Yes','StreamingMovies_No','StreamingMovies_Yes',
'Contract_One','Contract_Two','PaperlessBilling_Yes','PaymentMethod_Credit','PaymentMethod_Electronic','PaymentMethod_Mailed']
X = pima[feature_cols] # Features
y = pima.label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())



# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))   


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
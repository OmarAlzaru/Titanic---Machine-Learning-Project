# Import Packages for Feature Engineering and Data Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import Packages for Machine Learning Modelling
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import Datasets
train = pd.read_csv(r"D:\AI\Predicting-Titanic-Survivors/Datasets/train.csv")
test = pd.read_csv(r"D:\AI\Predicting-Titanic-Survivors/Datasets/test.csv")

# Plot Target Distribution In Train Dataset
sns.countplot(data=train, x='Survived')
plt.ylabel('Frequency')
plt.xlabel('Survival Category')
plt.xticks([0.0, 1.0], ['Did Not Survive', 'Survived']);

# Create Wrangler Function to Preprocess and Clean Dataset
def wrangler (df):
    df = df.copy()
    # Replace Column Elements for Readability
    emb = {"Q" : "Queenstown" , "S" : "Southampton" , "C" : "Cherbourg" }
    df['Embarked'].replace(emb)
    
    cla = {1:"First Class", 2: "Second Class", 3 :"Third Class"}
    df['Pclass'].replace(cla).astype('object')

    # Transform Ticket into Lettered Column by Ticket Type (Numeric or Alphanumeric Ticket)
    def has_letters(string):
        return any(char.isalpha() for char in string)
    df['Ticket_Type'] = df['Ticket'].apply(has_letters).astype('object')
    
    # Separate Dataset Features into Categorical and Numeric Variable (Useful For Visualizations)
    num_cols = list(df.select_dtypes(exclude=['object']).columns)
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    
    # Encode the Categorical Features
    le = LabelEncoder()
    for col in cat_cols:
        le.fit(df[col])
        df[col] = le.transform(df[col])
      
    # Dropped Unneeded Columns After Feature Engineering
    drop_cols = []
    drop_col_list = ["Name", "Cabin", "Ticket"]
    for col in df.columns:
        if col in drop_col_list:
            drop_cols.append(col)
    
    df = df.drop(columns = drop_cols)
    return df

# Preprocess Test and Train Datasets
train = wrangler(train)
test = wrangler(test)

# Create Target, Features, and Train-Test Split
X_train = train.drop(['Survived', 'PassengerId' ], axis=1)
y_train = train['Survived']
X_test = test.drop(['PassengerId' ], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Prediction Using the Logistic Regression Model
# Create Pipeline With Scaler, Imputer and Model  
model = LogisticRegression(random_state= 42, max_iter=1000)
scaler = RobustScaler()
imputer = KNNImputer(n_neighbors=5)
pipeline = Pipeline(steps=[('i', imputer), ('s', scaler) , ('m', model)])

# Fit the Pipeline on Training Data
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_train, y_train)
print(f" Training Accuracy: {accuracy}")

# Make Survival Prediction in the Test Dataset and Save Results to Excel File
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print("Confusion Matrix:")
print(conf_matrix)
# Error Analysis
# Identify Misclassified Samples
misclassified_samples = X_test[y_test != predictions]

# Analyze Misclassifications
misclassified_predictions = predictions[y_test != predictions]
true_labels = y_test[y_test != predictions]

misclassified_data = pd.DataFrame({'Predicted': misclassified_predictions, 'True Label': true_labels})
misclassified_data = pd.concat([misclassified_data, misclassified_samples.reset_index(drop=True)], axis=1)

# Visualize Misclassifications
# For example, you can plot histograms or count plots of specific features for misclassified samples
plt.figure(figsize=(10, 6))
sns.countplot(data=misclassified_data, x='Sex', hue='Predicted', palette={0: 'red', 1: 'green'})
plt.xlabel('Feature')
plt.ylabel('Count')
plt.title('Misclassifications by Feature')
plt.legend(['Did Not Survive', 'Survived'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

model_output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
model_output.to_csv('C:/Users/Lenovo/Downloads/LRPredictions.csv', index=False)

# Check how many survivors/non-survivors the Model Predicted
unique, counts = np.unique(predictions, return_counts=True)
dict(zip(unique, counts))

# Save Model to a .pkl File 
pickle.dump(pipeline, open(r'D:\AI\Predicting-Titanic-Survivors/Models/LRModel.pkl', 'wb'))

# Prediction Using the Logistic Regression Cross Validation Model
# Create Pipeline With Scaler, Imputer and Model
model = LogisticRegressionCV(cv=5, max_iter = 1000, random_state=42, n_jobs=-1)
scaler = RobustScaler()
imputer = KNNImputer(n_neighbors=5)
pipeline = Pipeline(steps=[('i', imputer), ('s', scaler) , ('m', model)])

# Fit the Pipeline on Training Data
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_train, y_train)
print(f" Training Accuracy: {accuracy}")

# Make Survival Prediction in the Test Dataset and Save Results to Excel File
predictions = pipeline.predict(X_test)
model_output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
model_output.to_csv('C:/Users/Lenovo/Downloads/LRCVPredictions.csv', index=False)

# Check how many survivors/non-survivors the Model Predicted
unique, counts = np.unique(predictions, return_counts=True)
dict(zip(unique, counts))

# Save Model to a .pkl File 
pickle.dump(pipeline, open(r'D:\AI\Predicting-Titanic-Survivors/Models/LRCVModel.pkl', 'wb'))

# Prediction Using the Decision Tree Classifier
# Create Pipeline With Scaler, Imputer and Model
pipeline = Pipeline([('scaler', StandardScaler()),  ('imputer', KNNImputer(n_neighbors=5)), ('classifier', DecisionTreeClassifier(random_state=42))])

# Prepare Hyperparameter Tuning Dictionary
param_grid = {'classifier__criterion': ['gini', 'entropy'], 'classifier__max_depth': [None, 5, 10, 15],
              'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4]}

# Tune Hyperparameters 
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Select the Best Estimator to Fit Training Data
best_estimator = grid_search.best_estimator_
accuracy = best_estimator.score(X_train, y_train)
print(f" Training Accuracy: {accuracy}")

# Make Survival Prediction in the Test Dataset and Save Results to Excel File
predictions = best_estimator.predict(X_test)
model_output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
model_output.to_csv('C:/Users/Lenovo/Downloads/DTCPredictions.csv', index=False)

# Check how many survivors/non-survivors the Model Predicted
unique, counts = np.unique(predictions, return_counts=True)
dict(zip(unique, counts))

# Save Model to a .pkl File 
pickle.dump(pipeline, open(r'D:\AI\Predicting-Titanic-Survivors/Models/DTCModel.pkl', 'wb'))

# Prediction Using Random Forest Classifier
# Create Pipeline With Scaler, Imputer and Model
pipeline = Pipeline([('scaler', StandardScaler()),  ('imputer', KNNImputer(n_neighbors=5)), ('classifier', RandomForestClassifier(random_state=42))])

# Prepare Hyperparameter Tuning Dictionary
param_grid = {'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [None, 5, 10, 15],
              'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4], 'classifier__bootstrap': [True, False]}
# Tune Hyperparameters 
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
#grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Select the Best Estimator to Fit Training Data
best_estimator = grid_search.best_estimator_
accuracy = best_estimator.score(X_train, y_train)
print(f" Training Accuracy: {accuracy}")

# Make Survival Prediction in the Test Dataset and Save Results to Excel File
predictions = best_estimator.predict(X_test)
model_output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
model_output.to_csv('C:/Users/Lenovo/Downloads/RFCPredictions.csv', index=False)

# Check how many survivors/non-survivors the Model Predicted
unique, counts = np.unique(predictions, return_counts=True)
dict(zip(unique, counts))

# Save Model to a .pkl File 
pickle.dump(pipeline, open(r'D:\AI\Predicting-Titanic-Survivors/Models/RFCModel.pkl', 'wb'))

# Check Feature Importance of Best Model and Save Best Model for Web App

# Saving the Best Model to appmodel.pkl for Web App
pipeline = Pipeline([('scaler', StandardScaler()),  ('imputer', KNNImputer(n_neighbors=5)), ('classifier', DecisionTreeClassifier(random_state=42, criterion = 'entropy', max_depth = 5, min_samples_leaf = 2, min_samples_split = 2))])
pipeline.fit(X_train, y_train)
pickle.dump(pipeline, open(r"D:\AI\Predicting-Titanic-Survivors\appmodel.pkl", 'wb'))

# Extracting Feature Importance in the Best Model
decision_tree = pipeline.named_steps['classifier']
importances = decision_tree.feature_importances_
feature_names = X_train.columns
feature_importances = dict(zip(feature_names, importances))

# Visualization of the Most Important Determinants of Survival
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()
# Plotting Survival Distribution by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='Sex', hue='Survived', palette={0: 'red', 1: 'green'})
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Survival Distribution by Gender')
plt.legend(['Did Not Survive', 'Survived'])
plt.xticks([0.0, 1.0], ['Females', 'Males']);
plt.show()

# Counting Survivors and Non-survivors by Gender
survival_by_gender = train.groupby(['Sex', 'Survived']).size().unstack()
survival_by_gender.columns = ['Did Not Survive', 'Survived']
survival_by_gender.index = ['Female', 'Male']
print(survival_by_gender)
# Define age intervals
age_intervals = pd.cut(train['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80])

# Define a new column indicating whether each passenger is alive or not
train['Survival_Status'] = train['Survived'].map({0: 'Not Alive', 1: 'Alive'})

# Plotting Survival Distribution by Age Intervals
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x=age_intervals, hue='Survival_Status', palette={'Not Alive': 'red', 'Alive': 'green'})
plt.xlabel('Age Intervals')
plt.ylabel('Count')
plt.title('Survival Distribution by Age Intervals')
plt.legend(['Not Alive', 'Alive'])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()



# Define age intervals
age_intervals = pd.cut(train['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80])

# Calculate survival rate for each age group
age_group_survival = train.groupby(age_intervals,observed=False)['Survived'].mean().reset_index()

# Rename columns for clarity
age_group_survival.columns = ['Age Group', 'Survival Rate']

# Display the DataFrame
print(age_group_survival)

# Calculate survival rate for each passenger class
class_survival_rate = train.groupby('Pclass')['Survived'].mean().reset_index()

# Rename columns for clarity
class_survival_rate.columns = ['Passenger Class', 'Survival Rate']

# Display the DataFrame
print(class_survival_rate)

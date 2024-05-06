# Predicting-Titanic-Survivors
## Project Overview
This repository contains models for predicting the survivors of the titanic shipwreck disaster of 1912. The model results were submitted to the Kaggle "Getting Started with Prediction" Competition. As part of the competition, competitors are required to use machine learning model(s) of their choice in predicting which passengers survived the Titanic shipwreck of 1912. The resulting prediction models are then scored, based on their accuracy.

## Dataset For Analysis
The dataset used for this ML project was obtained from [Kaggle](https://www.kaggle.com/competitions/titanic) and has been split into test and train sections. The train data (train.csv) which contains 891 observations is used to model the predictions of survival and subsequently used on the test data (test.csv) which contains 418 observations. The titanic data contains passenger information like Name, Sex, Age, Passenger Class (1st, 2nd or 3rd), Number of Siblings/Spouses Aboard, Number of Parents/Children Aboard, Ticket Number, Passenger Fare (in British pound), Cabin, and the Port of Embarkation (Cherbourg, Queenstown or Southampton). Based on these features, the passengers either survive or not surviive after the shipwreck. 

## Python Packages and Modules Needed
It is worthy to note that the train-test-split method does not need to be imported from Scikit-Learn because the data has been split prior to the start of the task. However, to make the cleaning and feature engineering a blanket process, the train and test datasets are merged and later split using the train-test-split method (I think this is an easier alternative to working on the splits separarely or using a function to handle the cleaning and feature engineering of both datasets). 

[Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) are imported for Data Manipulation and Wrangling. The Machine Learning Models are imported from [Scikit-Learn](https://scikit-learn.org/). Each model's prediction is then saved as a .pkl file using the Pickle module. 

## Chronology 
- Import necessary libraries and datasets.
- Identify datatypes and make appropriate conversions.
- Preprocess dataset - deal with missing values and labels in numeric and categorical features respectively.
- Split data into the features and target.
- Training machine learning models on training dataset and check training accuracy.
- Tune Hyperparameters of the model (if necessary) as the default settings sometimes yield the best performance for the model.
- Use trained model on test dataset and make prediction.
- Save predictions to a desired file format. 
- Save the model by dumping into a pickle file.
- Check the most important features for best model. 
- Create a simple web page where users can get a titanic survival prediction based on features. 

## Model Results
**Logistic Regression Model** : The Logistic Regression Model yielded an accuracy of 80% with the training data. The model predicted that 36% of the passengers in the Test Dataset survived the shipwreck while 64% did not survive. The Kaggle submission for this model returned a 76.94% accuracy score. 

**Logistic Regression Cross Validation Model** : The Logistic Regression Cross Validation Model yielded an accuracy of 81% with the training data. The model predicted that 37% of the passengers in the Test Dataset survived the shipwreck while 63% did not survive. The Kaggle submission for this model returned a 77.51% accuracy score.

**Decision Tree Classifier** : The Decision Tree Classifier Model yielded an accuracy of 84% with the training data. The model predicted that 35% of the passengers in the Test Dataset survived the shipwreck while 65% did not survive. The Kaggle submission for this model returned a 78.95% accuracy score.

**Random Forest Classifier** : The Random Forest Classifier Tree Model yielded an accuracy of 90% with the training data. The model predicted that 35% of the passengers in the Test Dataset survived the shipwreck while 65% did not survive. However, the Kaggle submission for this model returned a 77.27% accuracy score.

## Using Best Model for Simple Web Application Deployment 
Of all four models deployed on the Titanic Shipwreck dataset, the Decision Tree Classifier performed best with some hyperparameter tuning. The model identifies Sex, Fare paid for the Trip and the Passenger Cabin Class as the most important determinants of survival, while the Point of Embarkment and Ticket Type were the least important determinants of survival. 

The model was then used to train the dataset in preparation for its deployment as a simple web application. The web application was deployed as a single landing page using [Streamlit](https://streamlit.io/). Users are able to toggle between several dataset features to predict their survival category, whether they "Survived" or "Did not Make it".

## Author(s)
- [**Abraham Ajibade**](https://www.linkedin.com/in/abraham-ajibade-759772117)
- [**Boluwtife Olayinka**](https://www.linkedin.com/in/ajibade-bolu/)

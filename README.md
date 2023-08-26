# Web-App-Titanic-Survival

<p align="center">

  Come and check your chances of surviving the titanic shipwreck in this web <a href="https://web-app-titanic-survival-a81fad2fbdde.herokuapp.com/">app
  </a>
</p>
<p align="center">
  <a href="https://web-app-titanic-survival-a81fad2fbdde.herokuapp.com/"> 
    <img src="https://user-images.githubusercontent.com/50509053/185045576-4d4f78ce-1ad5-49a5-9756-0393b9088cb7.png" alt="Logo" width=700 height=250>
  </a>
</p>

# Project Overview 
Predicting if you will survive the titanic or not

* Created an accurate model that can predict the probability of you surviving or not the shipwreck
* Enter your passenger details in the web app and find out
* Predicts correctly with an 84% accuracy
* Feature engineered the titles from the passenger names
* Feature engineered if the person is alone or not from the number of relatives present on the ship
* Cleaned the data, normalized and scaled it appropriately
* Optimized Naive bayes, Logistic regression, decision tree, k nearest neighbors, random forest, support vector machine, xtreme gradient boosting using ensembling methods to reach the best model.
* Finally a soft voting ensembling classifier achieved the best accuracy.



## Code and Resources Used 
**Python Version:** 3.10.5 <br>
**Packages:** pandas, numpy, sklearn, requests, dill, Flask, xgboost, gunicorn, matplotlib, seaborn


## Data from the passengers:

| Variable | Definition   | Key   |
|----------|--------------|-------|
|  survival| Survival     |0 = No, 1 = Yes|
|pclass    | Ticket class |1 = 1st, 2 = 2nd, 3 = 3rd|
|sex       | Sex|
|Age  |Age in years|
|sibsp|# of siblings / spouses aboard the Titanic|
|parch|# of parents / children aboard the Titanic|
|ticket|Ticket number|
|fare|Passenger fare|
|cabin|Cabin number|
|embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|

## EDA
After getting the data, I explored it and looked for correlations:

* Plotted the relationship between the features and the target variable
* Compared various features to one another
* Determined whether or not features are unbalanced. Checked if the target's classes are unbalanced
* Calculated correlations between the various columns

## Feature engineering and cleaning
The steps I took in this phase:

* Drop out PassengerId(irrelevant), Name(feature engineered), Ticket(irrelevant) and Cabin(irrelevant and too many NaNs).
* Create a new feature 'IsAlone'( = SibSp + Parch + 1) to indicate if a passenger is alone.
* Extract titles from Name.
* Imputation of missing values and normalization of numerical features
* Encode the categorical features.


## Model Building 

I just used 'Age', 'Fare', 'Pclass', 'Sex', 'Embarked', 'IsAlone', and 'Title' because according to EDA they were the most relevant. <br>
Created a numerical pipeline, then a categorical pipeline, and then united them.

<p align="left">
  <a href="https://did-you-survive-the-titanic.herokuapp.com/">
    <img src="https://user-images.githubusercontent.com/50509053/185141990-09febd51-e476-49ff-b641-c8f236865434.png" alt="Logo" width=450 height=250>
  </a>
</p>


Then I passed the features through the pipeline <br>
Then I applied several machine learning models to the data and computed their cross validation scores on a validation set
<p align="left">
  <a href="https://web-app-titanic-survival-a81fad2fbdde.herokuapp.com/">
    <img src="https://user-images.githubusercontent.com/50509053/185145577-7bdfff38-cc57-4a97-997c-c279feafb287.png" alt="Logo" width=700 height=750>
  </a>
</p>

I plotted the learning curves for all of those models to see who would overfit or underfit<br>
I finally settled on an SVM, tuned it using gridsearchcv and computed its accuracy


## Model Additional Ensemble Approaches
Here I used ensembling algorithms to try to improve the model(Due to computational reasons, I did not tune these ensemble models to their max. This explains why their accuracy may be lower than that of the tuned SVM model)

1) Experimented with a hard voting classifier of three estimators (KNN, SVM, RF) (81.4%)

2) Experimented with a soft voting classifier of three estimators (KNN, SVM, RF) (81.7%) (best performance in competition leaderboard)

3) Experimented with soft voting on all estimators performing better than 80% except xgb (KNN, RF, LR, SVC) (82.6%)

4) Experimented with soft voting on all estimators including XGB (KNN, SVM, RF, LR, XGB) (82.8%) (Best Performance)
<br>

<p align="center">
  <a href="https://did-you-survive-the-titanic.herokuapp.com/">Try the app</a>
</p>
<p align="center">
  <a href="https://web-app-titanic-survival-a81fad2fbdde.herokuapp.com/">
    <img src="https://user-images.githubusercontent.com/50509053/185149779-e30b649d-5bfd-4220-b51e-09a5c8359a70.png" alt="Logo" width=400 height=100>
  </a>
</p>







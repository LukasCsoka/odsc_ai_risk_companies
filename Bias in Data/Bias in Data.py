# -*- coding: utf-8 -*-

# -- Sheet --

# #  Bias in Data
# 
# The code that I’m providing has been built mainly upon the following sources:
#  
# - https://fairmlbook.org/
# - https://dalex.drwhy.ai/python-dalex-fairness.html
# - https://dalex.drwhy.ai/python/
# - https://www.kdnuggets.com/2020/12/machine-learning-model-fair.html
#  
# Unverified black box model is the path to the failure. Opaqueness leads to distrust. Distrust leads to ignoration. Ignoration leads to rejection.
# 
# The dalex package xrays any model and helps to explore and explain its behaviour, helps to understand how complex models are working. The main [Explainer](https://dalex.drwhy.ai/python/api/#dalex.Explainer) object creates a wrapper around a predictive model. Wrapped models may then be explored and compared with a collection of model-level and predict-level explanations. Moreover, there are fairness methods and interactive exploration dashboards available to the user.
# 
# The philosophy behind dalex explanations is described in the [Explanatory Model Analysis e-book](https://ema.drwhy.ai/).


# basic imports
import dalex as dx
import numpy as np

# import scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix

# To showcase the problem of fairness in AI, we will be using the German Credit Data dataset to assign risk for each credit-seeker.
# Information about this dataset can be found here: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
# 
# We will at first create a very simple ML model (decision tree) for the data. Additionally, we will use create random forest and logistic regression, but thes two models will be used later.
# 
# It is very important to avoid any bias, when a person applies for loan in bank, as nobody would want to be negatively affected, as well if the bank does not have reliable model, the bank can lose part of business or provide loans to people, that would not receive loans by unbiased models. 
# 
# The data we use for modeling is in the major part a reflection of the world it derives from. And as the world can be biased, so data and therefore model will likely reflect that. 


# load credit data
data = dx.datasets.load_german()

# risk is the target variable
features = data.drop(columns='risk')
labels = data.risk

# select few categorical and numerical features
categorical_features = ['sex', 'job', 'housing', 'saving_accounts', "checking_account", 'purpose']
numeric_features = ['credit_amount', 'age']

# create one hot encoder for categorical variables as transformer
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# scale numerical features as transformer
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)])


# create a pipeline, containing the above transformer and decision tree
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=7, random_state=42))
])

# train decision tree on this data
clf.fit(features, labels)

# train also random forest - it will be used later
clf_forest = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, max_depth=5))]).fit(features,labels)

# trian also logistic regression - it will be used later
clf_logreg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42))]).fit(features,labels)

# plot confusion matrix of decision tree
plot_confusion_matrix(clf, features, labels)

# We create an Explainer object to showcase dalex functionalities. Then we look at the overal performance of our model. Even through its simple, it is not bad. At first, we will use only decision tree.


exp = dx.Explainer(clf, features, labels)
exp.model_performance().result

# To check if the model is biased, we will use the fairness module from dalex. Checking if the model is fair should be straightforward. Apart from the dx.Explainer, we will need 2 parameters:
# 
# - protected - array-like of subgroups values that denote a sensitive attribute (protected variable) like sex, nationality etc. The fairness metrics will be calculated for each of those subgroups and compared.
# - privileged - a string representing one of the subgroups. It should be the one suspected of the most privilege.
# 
# The idea here is that ratios between scores of privileged and unprivileged metrics should be close to 1. The closer the more fair the model is. But to relax this criterion a little bit, it can be written more thoughtfully:
# 
# $\forall_{i \varepsilon \{a,b,...,z\}}  \epsilon  \frac{metrix_i}{metric_{privileged}} < \frac{1}{\epsilon}$
# 
# where the epsilon is a value between 0 and 1. It should be a minimum acceptable ratio. On default, it is 0.8, which adheres to [four-fifths rule (80% rule)](https://www.hirevue.com/blog/hiring/what-is-adverse-impact-and-why-measuring-it-matters) commonly used. Of course, a user may change this value to their needs.


# array with values like male_old, female_young, etc.
protected = data.sex + '_' + np.where(data.age < 25, 'young', 'old')

privileged = 'male_old'  # we assume, that older males are prviliged compared to young females, lets thest this hypothesis

fobject = exp.model_fairness(protected = protected, privileged = privileged)
fobject.fairness_check(epsilon = 0.8)

# This model should not be called fair. Generally, each metric should be between (epsilon, 1/epsilon). Metrics are calculated for each subgroup, and then their scores are divided by the score of the privileged subgroup. That is why we omit male_old in this method. When at least 2 metrics have scores ratio outside of the epsilon range, dalex declared this model unfair. In our case it cannot be decided automatically but the bias is visible and FPR (False Positive Rate) is especially important in case of risk assigning, so let's call our model unfair.
# 
# The bias was spotted in metric FPR, which is the False Positive Rate. The output above suggests that the model cannot be automatically approved (like said in the output above). So it is up to the user to decide. In my opinion, it is not a fair model. Lower FPR means that the privileged subgroup is getting False Positives more frequently than the unprivileged.


# ## Let's check more metrics
# 
# We get the information about bias, the conclusion, and metrics ratio raw DataFrame. There are metrics TPR (True Positive Rate), ACC (Accuracy), PPV (Positive Predictive Value), FPR (False Positive Rate), STP(Statistical parity). The metrics are derived from a confusion matrix for each unprivileged subgroup and then divided by metric values based on the privileged subgroup. 
# 
# The result attribute is metric_scores where each row is divided by row indexed with privileged (in this case male_old).


fobject.result # to see all scaled metric values

fobject.metric_scores # or unscaled ones

# ## Let's look at some plot (dalex uses plotly)
# 
# There are two bias detection plots available (however, there are more ways to visualize bias in the package)
# 
# - fairness_check— visualization of fairness_check() method
# - metric_scores— visualization of metric_scores attribute which is raw scores of metrics.
# 
# For fairness_check, if a bar reaches the red field, it means that for this metric model is exceeding the (epsilon, 1/epsilon) range. In this case the DecisionTreeClassifier has one NaN. In this case appropriate message is given (it can be disabled with verbose=False).
# 
# For metric_scores, vertical lines showcase the score of the privileged subgroup. Points closer to the line indicate less bias in the model.


fobject.plot()

fobject.plot(type = 'metric_scores')

# ## Multiple models
# 
# Let's now use also random forest and logistic regression results.


# create Explainer objects 
exp_forest  = dx.Explainer(clf_forest, features, labels, verbose = False)
exp_logreg  = dx.Explainer(clf_logreg, features, labels, verbose = False)

# create fairness explanations
fobject_forest = exp_forest.model_fairness(protected, privileged)
fobject_logreg = exp_logreg.model_fairness(protected, privileged)

# fairness check
fobject_forest.fairness_check(epsilon = 0.8)
fobject_logreg.fairness_check(epsilon = 0.8)

fobject.plot(objects=[fobject_forest, fobject_logreg])

fobject.plot(objects=[fobject_forest, fobject_logreg], type = "radar")

# ## Metrics


# 
# | Metric | Formula | Full name | fairness names while checking among subgroups |
# | ----------- | ----------- | ----------- | ----------- |
# | TPR | $\frac{TP}{TP+FN}$ | true positive rate | equal opportunity |
# | TNR | $\frac{TN}{TP+FP}$ | true negative rate |  |
# | PPV | $\frac{TP}{TP+FP}$ | positive predictive value |  |
# | NPV | $\frac{TN}{TN+FN}$ | negative predictive value |  |
# | FNR | $\frac{FP}{FN+TP}$ | false negative rate	 |  |
# | FPR | $\frac{FN}{FP+TN}$ | false positive rate	 | predictive equality |
# | FDR | $\frac{FP}{FP+TP}$ | false discovery rate |  |
# | FOR | $\frac{FN}{FN+TN}$ | false ommision rate |  |
# | TS |  $\frac{TP}{TP+FN+FP}$ | threat score |  |
# | STP | $\frac{TP+FP}{TP+FN+FP+TN}$ | statistical parity | statistical parity |
# | ACC | $\frac{TP+TN}{TP+FN+FP+TN}$ | accuracy | overall accuracy equality |
# | F1 | $2\cdot \frac{PPV*TPR}{PPV-TPR}$ | f1 score |  |



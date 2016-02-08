
# coding: utf-8

# # Detailed walk through the Airbnb Dataset

# ## Get the Setup out of the way
#
# * Setup matplotlib for inline plotting
# * Setup math-related, statistics, machine learning, DataFrames
# * airbnb_tools is a set up wrappers  to do some basic jobs written by myself

# In[63]:



# Imports
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from airbnb_tools import *
import math

from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer




# ## Read in the data and start some Cleaning
#
# * Read in with pandas
# * At first, I was limiting to those with bookings. Turns out that's the wrong way to do it, they want a non-booking to be included
# * Replace missing data with -1 (this is for quick analysis)

# In[46]:

# Read in the training data
train = pd.read_csv("data/train_users_2.csv")

# Start some preliminary cleaning of the data.
# Fill in missing data with -1 for now
train.fillna(-1, inplace=True)


# ## Create a function that takes two YYYY-MM-DD dates and calculates the difference between them.
#
# Note that we allow an optional keyword (noNeg) to force the minimum output to be 0. We do this because some of the inputs we're dealing with should always be > 0 and a negative is probably fubar.

# In[51]:

# Create some features from others that may be telling
def days_between(d1, d2, noNeg=False):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    outval = (d2-d1).days
    if (outval > 0 or noNeg == False):
        return outval
    else:
        return 0

# Where were no bookings made?
d1 = train["date_account_created"].values
d2 = train["date_first_booking"].values
elapsed = []
for ii in range(0, len(train)):
    if (d2[ii] != -1):
        elapsed.append(days_between(d1[ii],d2[ii], noNeg=True))
    else:
        elapsed.append(10000.)
# Add this to the DataFrame
train["elapsed_booking_time"] = elapsed


# ## Create another little function to convert the (numeric) time stamp keyword to useful values.

# In[52]:

# What does the timestamp_first_active say?
def tfa(x):
    """ Return YYYY-MM-DD format for timestamp formatted data."""
    output = []
    x = str(x)
    return str(x[:4]) + '-' + str(x[4:6]) + '-' + str(x[6:8])



# In[53]:

# Split the initial Time Stamp into useful bits. The creation delay is the time between
# Their first search (which could be first) to when they created their account.
tfa_year = []
tfa_month = []
tfa_day = []
creation_delay = []
tfa_vector = train["timestamp_first_active"].values
for ii in range(0, len(train)):
    tfa_out = tfa(tfa_vector[ii])
    creation_delay.append(days_between(tfa_out, d1[ii]))


train["creation_delay"] = creation_delay
train.drop(["timestamp_first_active", "date_account_created",
            "date_first_booking"], axis=1, inplace=True)


# ## Clean up out of whack age data and remove unneeded features
#
# * For now, limit age to 15 < age < 100. Set others to -1
# * Remove ID which isn't a tracer of anything in this data set.

# In[54]:

# Clean up Age a bit.  Assume anything with age < 18 and age > 100 are fubar and set to -1
train['age'][(train['age'] < 15) | (train['age'] > 100)] = -1

# ID is likely not super informative, so let's drop it
train.drop(['id'], inplace=True, axis=1)


# ## Define and split the categorical values you want to use in a fit.
#
# * for now, I'm fitting nearly everything that's left.
# * Note that gender has M/F/prefer not to answer/Unknown, so not truely degenerate
# * Some of these may end up going later, but for now, let's see what we can do with ever feature.
# * This likely *will* overfit the data
#
# ## Finally, pop the destination to y
#
# * Rather than the string labels for countries, let's go a value between 0 and max_coutnries
#

# In[55]:

# What Categorical Variables are we interested in?
categorical_variables = ['gender', 'language', 'signup_method',  'signup_flow',
                         'affiliate_channel', 'affiliate_provider',
                         'first_affiliate_tracked', 'signup_app',
                         'first_device_type', 'first_browser']
X = split_categorical_variables(train, categorical_variables)
y = X.pop("country_destination")
label_table = LabelEncoder()
y = label_table.fit_transform(y.values)


# # Let's try a gradiant boost classifier

# In[56]:

xgb_model = XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.1)
xgb_model.fit(X, y)


# ## How did we do?
#
# * To start, let's look at how well we did just predicting the final outcome



pred = xgb_model.predict_proba(X)

# Find the most probable country
best_country = [] # Not used for now
bestId = []
for i in range(len(pred)):
    bestId.append(np.argsort(pred[i])[::-1])
    best_country.append(label_table.inverse_transform(bestId[-1]))



# ## Make a scorer for the model
#
# Following that mentioned in the evaluation by the project

# In[92]:

def dcg_score(true, test):
    """ Calculate the discounted culumlative gain for first 5 entries."""

    order = np.argsort(test)[::-1] # Reverse sort model predictions
    true = np.take(true, order[:5])

    discounts = np.log2(np.arange(len(true))+2)
    return np.sum(2**true-1/discounts)

def ndcg_score(truth, pred):

    lb = LabelBinarizer()
    lb.fit(range(len(pred)+1))
    T = lb.transform(truth)

    scores = []

    for y_true, y_score in zip(T, pred):
        actual = dcg_score(y_true, y_score)
        best = dcg_score(y_true, y_true)
        score = float(actual)/float(best)
        scores.append(scores)

        return np.mean(scores)


ndcg_score(y[0], bestId[0])

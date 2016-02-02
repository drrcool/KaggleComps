#!/usr/bin/python
"""This module contains some useful tools for Random Forests."""

import pandas as pd


def numerical_variables(frame):
    """Return the indices for any numerical variables in a dataframe.

    This function takes a dataframe (from pandas for example) and identifies
    any numerical columns and returns an index of just those.
    """
    return list(frame.dtypes[frame.dtypes != "object"].index)


def split_categorical_variables(frame, categorical_variables):
    """Return a DataFrame with Specific Categorical Variables.

    Split each of the values appearing in categorical variables into
    independent features for each entry. For example, a categorical
    variable "Sex" with values "M" and "F" would be split into "Sex_M" and
    "Sex_F", allowing each to be used in Random Forest Regression.

    The original "Sex" keyword is removed.
    """
    for variable in categorical_variables:

        # Fill Missing data with Missing
        frame[variable].fillna("Missing", inplace=True)
        dummies = pd.get_dummies(frame[variable], prefix=variable)

        # Update frame to include dummies and drop original variable
        frame = pd.concat([frame, dummies], axis=1)
        frame.drop([variable], axis=1, inplace=True)
    return frame


def describe_categorical(X):
    """Return the results for categorical like .describe()."""
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))


def printDataFrame(X, max_rows=10):
    """Print full dataframe."""
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))


def graph_feature_importances(model, feature_names, autoscale=True,
                              headroom=0.05, width=10,
                              summarized_columns=None):
    """Graph feature importances of random forest.

    Adapted by code by Mike Bernico.

    Parameters:
    model - name of the model whose features will be plotted
    feature_names -- list of the names of the features
    autoscale -- automaticaally adjust the X axis to the
                 largest feature plus headroom
    headroom -- buffer to add when autoscaling
    width -- figure width in inches
    summarized_columns -- a list of columns prefixes to summarize on
    """
    if autoscale:
        x_scale = model.feature_importances_.max() + headroom
    else:
        x_scale = 1

    # Setup the dictionary
    feature_dict = dict(zip(feature_names, model.feature_importances_))

    # Take care of the summarized_columns case
    if summarized_columns:
        # Some dummy columns need to be summarized
        for col_name in summarized_columns:
            # Sum all the features that contain col_name store in temp
            sum_value = sum(x for i, x in
                            feature_dict.items() if col_name in i)

            # Now remove all keys that are part of the col name
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
            for i in keys_to_remove:
                feature_dict.pop(i)

            # Lastly, read the summarized field
            feature_dict[col_name] = sum_value
    results = pd.Series(list(feature_dict.values()),
                        index=list(feature_dict.keys()))
    results.sort_values(inplace=True)
    results.plot(kind='barh', figsize=(width, len(results)/4),
                 xlim=(0, x_scale))

"""
Module containing functions pertaining to feature-space projections related to network smoothing.
When the data matrix features don't overlap 100% with the smoothing graph nodes.

@jonathanronen 2016/6
"""

import numpy as np
import pandas as pd


def _guess_feature_axis_from_new_features(data, new_features):
    """
    Guess which axis of `data` is the feature axis (as opposed to samples) by overlapping with `new_features`.
    """
    rows_overlap = len(set(data.index) & set(new_features))
    cols_overlap = len(set(data.columns) & set(new_features))
    return 0 if rows_overlap > cols_overlap else 1

def project_to_network(data, new_features, missing_value=0, feature_axis='auto'):
    """
    Takes AxB expression_dataframe and projects it onto CxB dataframe where B is len(new_features).
    Values for genes in `new_features` that are not present in `expression_dataframe.index` are set to 0.

    Project data dataframe onto a different feature space defined by `new_features`. Features in `new_features` that are missing from `data` are set to `missing_value`,
    while features from `new_features` that exist in `data` are kept.

        `data`:          pd.DataFrame of data
        `new_features`:  list of features for `data` to be projected onto
        `missing_value`: value to fill for `new_features` that don't have a value in `data`
        `feature_axis`:  which axis of `data` is features; 0 for rows, 1 for columns, 'auto' for a guess
    """
    feature_axis = _guess_feature_axis_from_new_features(data, new_features) if feature_axis=='auto' else feature_axis
    if feature_axis == 1:
        data = data.T

    expression_in_new_space = pd.DataFrame(np.zeros((len(new_features), expression_dataframe.shape[1])), index=smoothing_genes, columns=expression_dataframe.columns)
    genes_in_both = list(set(expression_dataframe.index) & set(smoothing_genes))
    expression_in_new_space.loc[genes_in_both,:] = expression_dataframe.loc[genes_in_both,:]

    return expression_in_new_space if feature_axis==0 else expression_in_new_space.T

def _guess_feature_axis_from_shape(data):
    """
    Guesses that the feature axis is the bigger one (that we have more features than samples)
    """
    return np.argmax(data.shape)

def project_from_network_recombine(original_expression, smoothed_expression, feature_axis='auto'):
    """
    Projects `smoothed_expression` back to the row-space of `original_expression`.
    The output has the shape of `original_expression`, with rows that are present in `smoothed_expression` overwritten,
    and rows that are present in `smoothed_expression` but missing from `original_expression` retain their values
    from `original_expression`.
    """
    feature_axis = _guess_feature_axis_from_shape(original_expression) if feature_axis=='auto' else feature_axis
    if feature_axis == 1:
        original_expression = original_expression.T
        smoothed_expression = smoothed_expression.T

    expr_in_e_space_smoothed = original_expression.copy()
    genes_in_both = list(set(original_expression.index) & set(smoothed_expression.index))
    expr_in_e_space_smoothed.loc[genes_in_both,:] = smoothed_expression.loc[genes_in_both,:]

    return expr_in_e_space_smoothed if feature_axis==0 else expr_in_e_space_smoothed.T

def smooth_and_recombine(expression_dataframe, smoothing_function, smoothing_genes, feature_axis='auto'):
    """
    Performs network smoothing when data feature space and smoothing network nodes dont overlap perfectly

      1. project data onto feature space defined by smoothing kernel (in `smoothing_genes`)
      2. perform network smoothing (by `smoothing_function)
      3. project smoothed data back onto original space

    The function `smoothing_function` should take in the expression pd.DataFrame (genes X samples) and return a pd.DataFrame of the same dimensions.
    """
    expression_in_network_space = project_to_network(expression_dataframe, smoothing_genes, feature_axis=feature_axis)
    expr_in_network_space_smoothed = smoothing_function(expression_in_network_space)
    expr_in_e_space_smoothed = project_from_network_recombine(expression_dataframe, expr_in_network_space_smoothed, feature_axis=feature_axis)
    return expr_in_e_space_smoothed

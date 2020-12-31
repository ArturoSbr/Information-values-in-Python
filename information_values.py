import numpy as np
import pandas as pd

def iv_numeric(target, feature, feature_name='feature', bins=10, return_table=False):
    '''
    Calculates the information value of a single numeric feature.  Make sure that `target` is
    a BINARY column where `0` implies non-event and `1` implies event.
    
    Parameters
    ----------
            target : array_like
                Input array or object that can be converted to an array where the dependent variable is stored.
                The target must be binary, where `0` implies non-event and `1` implies event. Must be the same
                length as `feature`.
            feature : array_like
                Input array or object that can be converted to an array where the independent variable is stored.
                Must be the same length as `target`.
            feature_name : str, default 'feature'
                Name of the column (optional). It will default to `feature` when unspecified.
            bins : int, default `10`
                Number of equally-sized bins to split the distribution into. When `bins` is greater than the maximum
                number of bins that the distribution can be split into, `bins` will be automatically reduced to its
                gretest possible value according to `pandas.qcut`.
            return_table : boolean, default `False`
                If `return_table == True`, it will return a summary table with the results grouped by bins.
                If `return_table == False`, it will only return a list in the form `[feature, information value]`
    '''
    t = pd.DataFrame({feature_name:feature,'target':target})
    t['bin'] = pd.qcut(t[feature_name], bins, labels=False, duplicates='drop') + 1
    t = t.groupby('bin').agg({'target':['size','sum']}).reset_index()
    t.columns = ['bin','count','bads']
    t['goods'] = t['count'] - t['bads']
    t['bads_pct'] = t['bads'].div(t['bads'].sum())
    t['goods_pct'] = t['goods'].div((t['goods']).sum())
    t['woe'] = np.log(t['goods_pct'].div(t['bads_pct']))
    t['iv'] = (t['goods_pct'] - t['bads_pct']) * t['woe']
    if return_table:
        return t
    else:
        return [feature_name, t['iv'].sum()]

def iv_categorical(target, feature, feature_name='feature', return_table=False):
    '''
    Calculates the information value of a single categorical feature.  Make sure that `target` is
    a BINARY column where `0` implies non-event and `1` implies event.
    
    Parameters
    ----------
            target : array_like
                Input array or object that can be converted to an array where the dependent variable is stored.
                The target must be binary, where `0` implies non-event and `1` implies event. Must be the same
                length as `feature`.
            feature : array_like
                Input array or object that can be converted to an array where the independent variable is stored.
                Must be the same length as `target`.
            feature_name : str, default 'feature'
                Name of the column (optional). It will default to `feature` when unspecified.
            return_table : boolean, default `False`
                If `return_table == True`, it will return a summary table with the results grouped by bins.
                If `return_table == False`, it will only return a list in the form `[feature, information value]`
    '''
    t = pd.DataFrame({feature_name:feature, 'target':target})
    t = t.groupby(feature_name).agg({'target':['size','sum']}).reset_index()
    t.columns = ['cat','count','bads']
    t['goods'] = t['count'] - t['bads']
    t['bads_pct'] = t['bads'].div(t['bads'].sum())
    t['goods_pct'] = t['goods'].div((t['goods']).sum())
    t['woe'] = np.log(t['goods_pct'].div(t['bads_pct']))
    t['iv'] = (t['goods_pct'] - t['bads_pct']) * t['woe']
    t['iv'].sum()
    if return_table:
        return t
    else:
        return [feature_name, t['iv'].sum()]

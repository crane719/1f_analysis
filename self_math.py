import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

import utils

def calc_corrcoef_matdf(df, correct_column_name=None):
    warnings.simplefilter('ignore', RuntimeWarning)
    column_names=list(df.columns)
    # correct labelが最初に来るように
    if not correct_column_name is None:
        column_names.remove(correct_column_name)
        column_names=[correct_column_name]+column_names

    # corrcoef_matdf
    corrcoef_matdf=pd.DataFrame(columns=column_names, index=column_names)
    for i in range(len(column_names)):
        col_name=column_names[i]
        print("     "+col_name)
        for j in range(i, len(column_names)):
            row_name=column_names[j]
            x=list(df[col_name])
            y=list(df[row_name])
            value = np.corrcoef(x,y)[0, 1]
            corrcoef_matdf.at[col_name, row_name] = value
            corrcoef_matdf.at[row_name, col_name] = value
    warnings.resetwarnings()

    # nanを0で埋める
    corrcoef_matdf=corrcoef_matdf.fillna(0)
    return corrcoef_matdf

def priority(df, column_name):
    """
    key_dataとother_dataの各変数間の相関を計算し，相関の絶対値が大きい順にsortしたkey, valueを返す
    :param df: 分析対象のdataframe
    :param column_name: 分析対象dataframeのcolumn名
    """
    warnings.simplefilter('ignore', RuntimeWarning)
    columns=list(df.columns)
    columns.remove(column_name)

    result=pd.DataFrame(columns=columns,index=[column_name])
    x=list(df[column_name])
    for column in columns:
        y=list(df[column])
        value = np.corrcoef(x,y)[0, 1]
        result.at[column_name, column]=value
    result=result.fillna(0)
    warnings.resetwarnings()
    result_dict=result.to_dict(orient="list")

    # 相関を絶対値に
    keys=list(result_dict.keys())
    values=list(result_dict.values())
    values=[abs(value[0]) for value in values]

    # sort
    args=list(reversed(np.argsort(values)))
    keys=np.array(keys)[args]
    values=np.array(values)[args]
    return keys, values

def standarization_1d(x):
    x=np.array(x)
    return (x-np.average(x))/np.std(x)

def normalizatin_1d(x):
    x=np.array(x)
    min_x=min(x)
    max_x=max(x)
    return (x-min_x)/(max_x-min_x)

def rmse(x, x_pred):
    x=np.array(x)
    x_pred=np.array(x_pred)
    return np.sqrt(np.average((x-x_pred)**2))

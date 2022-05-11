import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
import sys
import pickle

IGNORE = ['TroponinI', 'Fibrinogen', 'EtCO2', 'Bilirubin_direct']

def SIRS(row):
    sirs = ['HR', 'Temp', 'PaCO2', 'Resp', 'WBC']
    counter_sirs = 0
    if row['HR'] > 90:
        counter_sirs += 1
    if row['Temp'] < 36 or row['Temp'] > 38:
        counter_sirs += 1
    if row['PaCO2'] < 32 or row['Resp'] > 20:
        counter_sirs += 1
    if row['WBC'] > 12000 or row['WBC'] < 4000:
        counter_sirs += 1
    return counter_sirs


def transform_data_with_SIRS(df):
    """
    Returns the whole df if the patient didn't have sepsis (and label 0), or the truncated df if the patient had sepsis
    (up until the first row with SepsisLabel=1, and label 1). In addition create new SIRS column - count the creiteria number
    """

    df['SIRS'] = df.apply(SIRS, axis=1)

    if df['SepsisLabel'].sum() == 0:
        return df.drop(columns='SepsisLabel').drop(columns=IGNORE), 0
    ind = df.SepsisLabel.where(df.SepsisLabel == 1).first_valid_index()
    return df.drop(columns='SepsisLabel').drop(columns=IGNORE).loc[0:ind], 1

def main():
    args = sys.argv[1:]
    path = args[0]

    X_train = []
    Y_train = []
    for file in os.listdir(f"{path}/"):
        df = pd.read_csv(f"{path}/{file}", sep='|')
        transformed_data, label = transform_data_with_SIRS(df)
        transformed_data['ICULOS'] = transformed_data['ICULOS'].max()
        mean_data = pd.DataFrame(transformed_data.mean()).T
        X_train.append(mean_data)
        Y_train.append(label)
    all_data_means = pd.concat(X_train).reset_index(drop=True)
    all_data_means = all_data_means.drop(columns='Unit2')
    most_freq_unit = all_data_means['Unit1'].value_counts().index[0]
    all_data_means['Unit1'] = all_data_means['Unit1'].fillna(most_freq_unit)
    all_data_means = all_data_means.fillna(all_data_means.mean())

    xgboost = XGBClassifier(n_estimators=500, use_label_encoder=False, scale_pos_weight=12,
                            max_depth=8,verbosity=1, eval_metric='error', max_delta_step=0.15,
                            subsample=None, alpha=0)

    xgboost.fit(all_data_means, Y_train)
    pickle.dump(xgboost, open("xgboost.pkl", "wb"))




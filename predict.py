import pandas as pd
import os
import numpy as np
import xgboost
import sys
import pickle

MOST_FREC_UNIT = 0.0
MEAN_IMPUTATION = pd.Series({'HR': 83.74764245904625,
 'O2Sat': 97.15269764896851,
 'Temp': 36.83867106385311,
 'SBP': 123.20322757968626,
 'MAP': 82.5323640644965,
 'DBP': 64.09616871048506,
 'Resp': 18.551032234670384,
 'BaseExcess': -0.483633605194992,
 'HCO3': 24.386175146999047,
 'FiO2': 0.6370687315975487,
 'pH': 7.381403041673607,
 'PaCO2': 40.99927307482935,
 'SaO2': 93.59674883146866,
 'AST': 143.81223995273913,
 'BUN': 21.73698172760817,
 'Alkalinephos': 96.12688761747974,
 'Calcium': 8.03576023629189,
 'Chloride': 105.59881595132249,
 'Creatinine': 1.4097833907422084,
 'Glucose': 131.96420302959567,
 'Lactate': 2.1828198722223506,
 'Magnesium': 2.0214482186211553,
 'Phosphate': 3.5397073435911834,
 'Potassium': 4.09877800768855,
 'Bilirubin_total': 1.4306505958431814,
 'Hct': 32.04417989068547,
 'Hgb': 10.716970207417775,
 'PTT': 36.95085542466277,
 'WBC': 11.013110751216391,
 'Platelets': 207.232551034029,
 'Age': 61.668051999999754,
 'Gender': 0.5555,
 'Unit1': 0.30385,
 'HospAdmTime': -50.9751960000032,
 'ICULOS': 38.2353,
 'SIRS': 0.6444512630622778})
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

    X_test = []
    Y_test = []
    patient_list = []
    for file in os.listdir(f"{path}/"):
        df = pd.read_csv(f"{path}/{file}", sep='|')
        patient_list.append(file)
        transformed_data, label = transform_data_with_SIRS(df)
        transformed_data['ICULOS'] = transformed_data['ICULOS'].max()
        mean_data = pd.DataFrame(transformed_data.mean()).T
        X_test.append(mean_data)
        Y_test.append(label)

    all_data_means_test = pd.concat(X_test).reset_index(drop=True)
    all_data_means_test = all_data_means_test.drop(columns='Unit2')
    all_data_means_test['Unit1'] = all_data_means_test['Unit1'].fillna(MOST_FREC_UNIT)
    all_data_means_test = all_data_means_test.fillna(MEAN_IMPUTATION)

    model = xgboost.XGBClassifier()
    model = pickle.load(open("xgboost.pkl", "rb"))

    y_pred = model.predict(all_data_means_test)

    patient_list_id = [int(i.replace('patient_', '').replace('.psv', '')) for i in patient_list]

    df = pd.DataFrame({'Id': patient_list_id, 'SepsisLabel': y_pred})
    df.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
    main()
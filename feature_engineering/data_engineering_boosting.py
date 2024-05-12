import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from feature_engineering.data_engineering import calcu_trading_entropy


def base_load_data(args: dict):
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    if os.path.exists("data/tef_1d.npy"):
        return
    features, labels = span_data_1d(feat_df)
    trf, tef, trl, tel = train_test_split(features, labels, stratify=labels, train_size=train_size, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args['testfeature'], args['trainlabel'], args['testlabel']
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def span_data_1d(
        data: pd.DataFrame,
        time_windows: list = [1, 3, 5, 10, 20, 50, 100, 500]
) -> (np.ndarray, np.ndarray):
    """transform transaction record into feature list

    Args:
        df (pd.DataFrame): transaction records
        time_windows (list): feature generating time length

    Returns:
        np.ndarray: array of features
    """
    data = data[data['Labels'] != 2]
    # data = data[data['Amount'] != 0]

    nume_feature_ret, label_ret = [], []
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record['Source']
        feature_of_one_record = []

        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(
                                                     row_idx - time_span):row_idx + 1, :]
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # AvgAmountT
            feature_of_one_timestamp.append(
                prev_records['Amount'].sum() / time_span)
            # TotalAmountTs
            feature_of_one_timestamp.append(prev_records['Amount'].sum())
            # BiasAmountT
            feature_of_one_timestamp.append(
                record['Amount'] - feature_of_one_timestamp[0])
            # NumberT
            feature_of_one_timestamp.append(len(prev_records))
            # MostCountryT/MostTerminalT -> no data for these items

            # MostMerchantT
            # feature_of_one_timestamp.append(prev_records['Type'].mode()[0] if len(prev_records) != 0 else 0)

            # TradingEntropyT ->  TradingEntropyT = EntT − NewEntT
            old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
            new_ent = calcu_trading_entropy(prev_and_now_records[['Amount', 'Type']])
            feature_of_one_timestamp.append(old_ent - new_ent)

            feature_of_one_record.append(feature_of_one_timestamp)
        flat_list = [
            x
            for xs in feature_of_one_record
            for x in xs
        ]
        nume_feature_ret.append(flat_list)
        label_ret.append(record['Labels'])

    return np.array(nume_feature_ret), np.array(label_ret, dtype=np.uint32)

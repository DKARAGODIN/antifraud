from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from feature_engineering.data_engineering_boosting import base_load_data


def xgboost_main(args):
    base_load_data(args)
    train_feature = np.load(args['trainfeature'], allow_pickle=True)
    train_label = np.load(args['trainlabel'], allow_pickle=True)
    test_feature = np.load(args['testfeature'], allow_pickle=True)
    test_label = np.load(args['testlabel'], allow_pickle=True)

    model = XGBClassifier()
    model.fit(train_feature, train_label)
    pred = model.predict(test_feature)

    true = test_label
    pred = np.array(pred)
    print(f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")

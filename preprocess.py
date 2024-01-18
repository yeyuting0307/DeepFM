
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import sys

if sys.argv[0] == 'torch_train.py':
    from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
    print("deepctr_torch loaded")
else:
    from deepctr.feature_column import  SparseFeat, DenseFeat, get_feature_names
    print("deepctr(tf) loaded")

def deepctr_transformer(train, test):
    ''' 
    train : ["id", "null", "bin_*", "nom_*", "ord_*", "intvl_*", "target"]
    test  : ["id", "null", "bin_*", "nom_*", "ord_*", "intvl_*"]
    '''
    test['target'] = -1
    data = pd.concat([train, test]).reset_index(drop=True)
    data['null'] = data.isna().sum(axis=1)

    nominal_feat = train.filter(regex=r'^nom_', axis=1).columns.to_list()
    ordinal_feat = train.filter(regex=r'^ord_', axis=1).columns.to_list()
    binary_feat = train.filter(regex=r'^bin_', axis=1).columns.to_list()
    interval_feat = train.filter(regex=r'^intvl_', axis=1).columns.to_list()

    sparse_features = binary_feat + nominal_feat + ordinal_feat
    dense_features = interval_feat

    # fillna
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )


    if sparse_features:
        for feat in sparse_features:
            lbe = LabelEncoder() # category -> numeric factor 
            data[feat] = lbe.fit_transform(data[feat].fillna('-1').astype(str).values)
    if dense_features:
        scaler = MinMaxScaler(feature_range=(0,1))
        data[dense_features] = scaler.fit_transform(data[dense_features])

    train = data[data.target != -1].reset_index(drop=True)
    test  = data[data.target == -1].reset_index(drop=True)


    fixlen_feature_columns = [
        SparseFeat(feat, data[feat].nunique()) for feat in sparse_features
    ]+ [
        DenseFeat(feat, 1) for feat in dense_features
    ]
    
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns


    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    feature_map = {
        "feature_names" : feature_names,
        "sparse_features" : sparse_features,
        "dense_features" : dense_features,
        "dnn_feature_columns" : dnn_feature_columns,
        "linear_feature_columns" : linear_feature_columns
    }
    

    return train, test, feature_map

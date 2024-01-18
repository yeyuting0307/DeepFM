#%%
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


from deepctr.models import DeepFM
from preprocess import deepctr_transformer
from tf_util.lr_scheduler import CyclicLR
from tf_util.auc import auc

## =================== Data Load =================== ##
''' 
    train : ["id", "bin_*", "nom_*", "ord_*", "intvl_*", "target"]
    test  : ["id", "bin_*", "nom_*", "ord_*", "intvl_*"]
'''

train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')


## =================== Feature Process =================== ##
train, test, feature_map = deepctr_transformer(train, test)

feature_names = feature_map.get("feature_names")
sparse_features = feature_map.get("sparse_features")
dense_features = feature_map.get("dense_features")
dnn_feature_columns = feature_map.get("dnn_feature_columns")
linear_feature_columns = feature_map.get("linear_feature_columns")


## ================ DeepFM ================ ##
target = ['target']
N_Splits = 5
Verbose = 0
Epochs = 2
SEED = 2023
Batch_S_T = 128
Batch_S_P = 512


oof_pred_deepfm = np.zeros((len(train), ))
y_pred_deepfm = np.zeros((len(test), ))
skf = StratifiedKFold(
    n_splits=N_Splits, 
    shuffle=True, 
    random_state=SEED
)

for fold, (tr_ind, val_ind) in enumerate(skf.split(train, train[target])):
    X_train, X_val = train[sparse_features + dense_features].iloc[tr_ind], train[sparse_features + dense_features].iloc[val_ind]
    y_train, y_val = train[target].iloc[tr_ind], train[target].iloc[val_ind]

    train_model_input = {name:X_train[name] for name in feature_names}
    val_model_input = {name:X_val[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    model = DeepFM(
        linear_feature_columns, 
        dnn_feature_columns, 
        dnn_hidden_units=(256, 256), 
        dnn_dropout=0.0, 
        dnn_activation='relu', 
        dnn_use_bn=False, 
        task='binary'
    )
    model.compile('adam', loss='binary_crossentropy', metrics=[auc], )

    es = callbacks.EarlyStopping(
        monitor='val_auc', 
        min_delta=0.001, 
        patience=3, 
        verbose=Verbose, 
        mode='max', 
        baseline=None, 
        restore_best_weights=True
    )
    sb = callbacks.ModelCheckpoint(
        './nn_model.h5', 
        save_weights_only=True, 
        save_best_only=True, 
        verbose=Verbose
    )
    clr = CyclicLR(
        base_lr=1e-7,
        max_lr = 1e-4, 
        step_size= int(1.0*(test.shape[0])/(Batch_S_T*4)) , 
        mode='exp_range', 
        gamma=1.0, 
        scale_fn=None, 
        scale_mode='cycle'
    )
    
    history = model.fit(
        train_model_input, 
        y_train,
        validation_data=(val_model_input, y_val),
        batch_size=Batch_S_T, 
        epochs=Epochs, 
        verbose=Verbose,
        callbacks=[es, sb, clr]
    )
    
    model.load_weights('./nn_model.h5')
    val_pred = model.predict(val_model_input, batch_size=Batch_S_P)
    print(f'<Val> AUC [fold {fold+1}]: {round(roc_auc_score(y_val, val_pred), 5)}')

    oof_pred_deepfm[val_ind] = val_pred.ravel()
    y_pred_deepfm += model.predict(
        test_model_input, 
        batch_size=Batch_S_P
    ).ravel() / (N_Splits)

    K.clear_session()

#%%
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
from preprocess import deepctr_transformer
from deepctr_torch.models import DeepFM
#%%
## =================== Data Load =================== ##
''' 
    train : ["id", "bin_*", "nom_*", "ord_*", "intvl_*", "target"]
    test  : ["id", "bin_*", "nom_*", "ord_*", "intvl_*"]
'''

train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')


# %%

train, test, feature_map = deepctr_transformer(train, test)

feature_names = feature_map.get("feature_names")
sparse_features = feature_map.get("sparse_features")
dense_features = feature_map.get("dense_features")
dnn_feature_columns = feature_map.get("dnn_feature_columns")
linear_feature_columns = feature_map.get("linear_feature_columns")


# %%

N_Splits = 5
SEED = 2023

skf = StratifiedKFold(
    n_splits=N_Splits, 
    shuffle=True, 
    random_state=SEED
)

oof_pred_deepfm = np.zeros((len(train), ))
y_pred_deepfm = np.zeros((len(test), ))

target = ['target']
Epochs = 2
Verbose = 0
Batch_S_T = 128
Batch_S_P = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for fold, (tr_ind, val_ind) in enumerate(skf.split(train, train[target])):
    X_train, X_val = train[sparse_features + dense_features].iloc[tr_ind], train[sparse_features + dense_features].iloc[val_ind]
    y_train, y_val = train[target].iloc[tr_ind].to_numpy(), train[target].iloc[val_ind].to_numpy()

    train_model_input = {name:X_train[name] for name in feature_names}
    val_model_input = {name:X_val[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    model = DeepFM(linear_feature_columns,
                   dnn_feature_columns,
                   use_fm=True,
                     dnn_hidden_units=(256, 256),
                     dnn_dropout=0.0,
                     dnn_activation='relu',
                     dnn_use_bn=False,
                     task='binary',
                     device=device
        )
    
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['binary_crossentropy','auc']
    ) # ["binary_crossentropy", "auc", "mse", "accuracy"]

    model.train()
    history = model.fit(
        train_model_input, 
        y_train,
        validation_data=(val_model_input, y_val),
        batch_size=Batch_S_T, 
        epochs=Epochs, 
        verbose=Verbose,
    )
    print('<train loss>\n')
    [print(f"{round(h, 4)}\n") for h in history.history.get('loss', [])]
    print('<val_binary_crossentropy>\n')
    [print(f"{round(h, 4)}\n") for h in history.history.get('val_binary_crossentropy', [])]
    print('<val_auc>\n')
    [print(f"{round(h, 4)}\n") for h in history.history.get('val_auc', [])]

    
    torch.save(model.state_dict(), './nn_model.pth')

    # =================== Evaluation =================== #
    model.load_state_dict(torch.load('./nn_model.pth'))
    model.eval()
    with torch.no_grad():
        val_pred = model.predict(val_model_input, batch_size=Batch_S_P)
        print(f'<Val> Check AUC [fold {fold+1}]: {round(roc_auc_score(y_val, val_pred), 5)}')

        oof_pred_deepfm[val_ind] = val_pred.ravel()
        y_pred_deepfm += model.predict(
            test_model_input, 
            batch_size=Batch_S_P
        ).ravel() / (N_Splits)

    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
# %%

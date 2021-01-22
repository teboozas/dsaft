import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
from torch import Tensor
import torchtuples as tt
import argparse
from pycox.datasets import metabric, gbsg, support, flchain, nwtco
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import warnings
from lifelines import KaplanMeierFitter

import wandb
import pdb
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
import warnings
import math
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from practical_kkbox import MixedInputMLP
from loss import DSAFTRankLoss,DSAFTMAELoss,DSAFTRMSELoss,DSAFTNKSPLLossNew

def get_surv(model, x_test, timegrid = None):
    '''
    model: PyCox model class or compatibles
    x_test: covariate dataset to compute survival estimates
    timegrid: option to set upperbound of time grid to "Y" of training dataset
    '''
    warnings.simplefilter(action='ignore', category = FutureWarning)
    warnings.simplefilter(action='ignore', category = RuntimeWarning)
    
    x_train, target = model.training_data
    y_train, delta_train = target
    
    # compute residual from training data
    # exp_residual = np.nan_to_num(np.exp(np.log(y_train) - model.predict(x_train).reshape(-1)))

    # compute exp(-theta) from test data to evaluate accelerating component
    exp_predict = np.nan_to_num(np.exp(-model.predict(x_test)).reshape(-1))
    
    # estimate cumulative baseline hazard function
    # based on training dataset
    H = NelsonAalenFitter().fit(y_train, event_observed = delta_train).cumulative_hazard_
    
    # extract timegrid and estimated hazards
    if timegrid == "train":
        max_time = y_train.max()
    else:
        max_time = max(H.index)
    
    if H.shape[0] * exp_predict.shape[0] >= 6 * 10e7:        
        l = round(5 * 10e7 / exp_predict.shape[0])
        time_grid = np.quantile(a = H.loc[H.index <= max_time].index.values,
                                q = [i / l for i in range(l + 1)],
                                interpolation = 'nearest')
    else:
        time_grid = H.loc[H.index <= max_time].index.values
    
    H_base = H.loc[time_grid].values.reshape(-1)
    
    h_base = H_base[1:] - H_base[:-1]
    h_base = np.repeat(h_base.reshape(-1, 1), exp_predict.shape[0], axis = 1)
    
    # evaluate conditional cumulative hazard estimates
    # based on test dataset
    surv = pd.DataFrame(np.exp(-np.cumsum(h_base * exp_predict, axis = 0)),
                        index = time_grid[1:],
                        columns = [i for i in range(exp_predict.shape[0])])
    
    surv.index.names = ["duration"]
    
    return surv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival analysis configuration')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='metabric')
    parser.add_argument('--loss', type=str, default='kspl')
    parser.add_argument('--an', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    

    parser.add_argument('--num_nodes', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_BN', action='store_true')
    parser.add_argument('--use_output_bias', action='store_true')
    
    args = parser.parse_args()

    print(args)


    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # Data preparation ==============================================================
    if args.dataset=='metabric':
        df_train = metabric.read_df()
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        cols_categorical = []
        num_embeddings = [len(pd.concat([df_train,df_val,df_test])[cat].unique()) for cat in cols_categorical]
        embedding_dims = [math.ceil(n_emb/2) for n_emb in num_embeddings]
    elif args.dataset=='gbsg':
        df_train = gbsg.read_df()
        cols_standardize = ["x3", "x4", "x5", "x6"]
        cols_leave = ["x0", "x2"]
        cols_categorical = ["x1"]
        num_embeddings = [len(pd.concat([df_train,df_val,df_test])[cat].unique()) for cat in cols_categorical]
        embedding_dims = [math.ceil(n_emb/2) for n_emb in num_embeddings]
    elif args.dataset=='support':
        df_train = support.read_df()
        cols_standardize =  ["x0", "x3", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
        cols_leave = ["x1", "x4", "x5"]
        cols_categorical =  ["x2", "x6"]
        num_embeddings = [len(pd.concat([df_train,df_val,df_test])[cat].unique()) for cat in cols_categorical]
        embedding_dims = [math.ceil(n_emb/2) for n_emb in num_embeddings]
    elif args.dataset=='flchain':
        df_train = flchain.read_df()
        df_train.columns =  ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "duration", "event"]
        cols_standardize =  ["x0", "x3", "x4", "x6"]
        cols_leave = ["x1", "x7"]
        cols_categorical = ["x2", "x5"]
        num_embeddings = [len(pd.concat([df_train,df_val,df_test])[cat].unique()) for cat in cols_categorical]
        embedding_dims = [math.ceil(n_emb/2) for n_emb in num_embeddings]
    elif args.dataset=='kkbox':
        from kkbox import _DatasetKKBoxChurn
        kkbox_v1 = _DatasetKKBoxChurn()
        try:
            df_train = kkbox_v1.read_df(subset='train')
            df_val = kkbox_v1.read_df(subset='val')
            df_test = kkbox_v1.read_df(subset='test')
        except:
            kkbox_v1.download_kkbox()
            df_train = kkbox_v1.read_df(subset='train')
            df_val = kkbox_v1.read_df(subset='val')
            df_test = kkbox_v1.read_df(subset='test')
        cols_standardize = ['n_prev_churns', 'log_days_between_subs', 'log_days_since_reg_init' ,'age_at_start', 'log_payment_plan_days', 'log_plan_list_price', 'log_actual_amount_paid']
        cols_leave =['is_auto_renew', 'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns']
        cols_categorical = ['city', 'gender', 'registered_via']
        # pdb.set_trace()
        # num_embeddings = [len(pd.concat([df_train,df_val,df_test])[cat].unique())+1 for cat in cols_categorical]
        # num_embeddings.append(pd.concat([df_train,df_val,df_test])[cols_categorical[-1]].unique())
        # embedding_dims = [math.ceil(n_emb/2) for n_emb in num_embeddings]

        # for cat in cols_categorical:
        #     for i, value in enumerate(pd.concat([df_train,df_val,df_test])[cat].unique()):
        #         df_train[cat].replace(value, i, inplace=True)
        #         df_val[cat].replace(value, i, inplace=True)
        #         df_test[cat].replace(value, i, inplace=True)
                # df_train["gender"].replace('male', 1, inplace=True)
                # df_train["gender"].replace('female', 2, inplace=True)
                # df_train["gender"].replace(np.NaN, 0, inplace=True)
                
                # df_val["gender"].replace('male', 1, inplace=True)
                # df_val["gender"].replace('female', 2, inplace=True)
                # df_val["gender"].replace(np.NaN, 0, inplace=True)
                
                # df_test["gender"].replace('male', 1, inplace=True)
                # df_test["gender"].replace('female', 2, inplace=True)
                # df_test["gender"].replace(np.NaN, 0, inplace=True)
        # pdb.set_trace()

    if not (args.dataset == 'kkobx'):
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)


    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize + leave)
    x_mapper_long = DataFrameMapper(categorical)

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))

    x_train = x_fit_transform(df_train)
    x_val = x_transform(df_val)
    x_test = x_transform(df_test)
    num_embeddings = x_train[1].max(0) + 1
    embedding_dims = num_embeddings // 2

    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = x_val, y_val

    def replace_zero(duration):
        return np.where(duration <= 0.0, duration + np.sort(np.unique(duration))[1], duration)

    # log-transformed time-to-event variables with replacement of zero-valued instances
    train_log_replace = np.log(replace_zero(y_train[0])).reshape(-1, 1)
    val_log_replace = np.log(replace_zero(y_val[0])).reshape(-1, 1)
    test_log_replace = np.log(replace_zero(durations_test)).reshape(-1, 1)

    # standardizer trained with training dataset
    scaler_train = StandardScaler().fit(train_log_replace)

    # scaled time-to-event datasets for consistent training
    y_train_transformed = (np.exp(scaler_train.transform(train_log_replace).reshape(-1)), y_train[1])
    y_val_transformed = (np.exp(scaler_train.transform(val_log_replace).reshape(-1)), y_val[1])
    val_transformed = x_val, y_val_transformed 
    durations_test_transformed = np.exp(scaler_train.transform(test_log_replace).reshape(-1))


    # Model preparation =============================================================    
    in_features = x_train[0].shape[1]
    num_nodes = [args.num_nodes]* args.num_layers
    out_features = 1
    batch_norm = args.use_BN
    dropout = args.dropout
    output_bias = args.use_output_bias
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
    #                             dropout, output_bias=output_bias)
    net = MixedInputMLP(in_features, num_embeddings, embedding_dims, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
    net = net.to(device)
    model = CoxPH(net, optimizer=tt.optim.AdamWR(lr=args.lr, decoupled_weight_decay=args.weight_decay),device=device)
    

    wandb.init(project='new_'+args.dataset, 
            group=args.loss,
            name=f'L{args.num_layers}N{args.num_nodes}D{args.dropout}W{args.weight_decay}B{args.batch_size}',
            config=args)

    wandb.watch(net)

    # Loss configuration ============================================================

    if args.loss =='rank':
        model.loss = DSAFTRankLoss()
    elif args.loss == 'mae':
        model.loss = DSAFTMAELoss()
    elif args.loss == 'rmse':
        model.loss = DSAFTRMSELoss()
    elif args.loss =='kspl_new':
        model.loss = DSAFTNKSPLLossNew(args.an, args.sigma)

    # Training ======================================================================
    batch_size = args.batch_size
    # lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
    # best = lrfinder.get_best_lr()

    model.optimizer.set_lr(args.lr)
    
    epochs = args.epochs
    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    verbose = True
    log = model.fit(x_train, y_train_transformed, batch_size, epochs, callbacks, verbose, val_data = val_transformed, val_batch_size = batch_size)

    # Evaluation ===================================================================
    surv = get_surv(model, x_test)
    ev = EvalSurv(surv, durations_test_transformed, events_test, censor_surv='km')
    # ctd = ev.concordance_td()
    ctd = concordance_index(event_times = durations_test_transformed,
                            predicted_scores = model.predict(x_test).reshape(-1),
                            event_observed = events_test)
    time_grid = np.linspace(durations_test_transformed.min(), durations_test_transformed.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    nbll = ev.integrated_nbll(time_grid)
    val_loss = min(log.monitors['val_'].scores['loss']['score'])
    wandb.log({'val_loss':val_loss,
                'ctd':ctd,
                'ibs':ibs,
                'nbll':nbll})
    wandb.finish()


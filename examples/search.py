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
import numpy as np
import pandas as pd
import warnings

from loss import DSAFTRankLoss,DSAFTMAELoss,DSAFTRMSELoss,DSAFTNKSPLLoss,DSAFTNKSPLLossNew

def get_surv(model, x_test, timegrid = "train"):
    '''
    model: PyCox model class or compatibles
    x_test: covariate dataset to compute survival estimates
    '''
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    x_train, target = model.training_data
    y_train, delta_train = target
    
    # compute residual from training data
    exp_residual = np.exp(np.log(y_train) - model.predict(x_train).reshape(-1))

    # compute exp(-theta) from test data to evaluate accelerating component
    exp_predict = np.exp(-model.predict(x_test)).reshape(-1)
    
    # estimate cumulative baseline hazard function
    # based on training dataset
    H = NelsonAalenFitter().fit(exp_residual, event_observed = delta_train).cumulative_hazard_
    
    # extract timegrid and estimated hazards
    time_grid = H.index.to_numpy()[1:]
    H_base = H.values.reshape(-1)
    
    h_base = H_base[1:] - H_base[:-1]
    h_base = np.repeat(h_base.reshape(-1, 1), exp_predict.shape[0], axis = 1)
    
    # evaluate conditional cumulative hazard estimates
    # based on test dataset
    surv = pd.DataFrame(np.exp(-np.cumsum(h_base * exp_predict, axis = 0)),
                        index = time_grid, columns = [i for i in range(exp_predict.shape[0])])
    surv.index.names = ["duration"]
    
    # set upperbound of time grid to "Y" of training dataset
    # (to be comparable to survival predictions from PyCox models)
    if timegrid == "train":
        surv = surv.loc[surv.index <= y_train.max()]
    
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
    elif args.dataset=='gbsg':
        df_train = gbsg.read_df()
        cols_standardize = ["x3", "x4", "x5", "x6"]
        cols_leave = ["x0", "x1", "x2"]
    elif args.dataset=='support':
        df_train = support.read_df()
        cols_standardize = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
        cols_leave = ["x1", "x2", "x3", "x4", "x5", "x6"]
    elif args.dataset=='flchain':
        df_train = flchain.read_df()
        df_train.columns = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "duration", "event"]
        cols_standardize = ["x0", "x2", "x3", "x4", "x6"]
        cols_leave = ["x1", "x5", "x7"]
    elif args.dataset=='nwtco':
        df_train = nwtco.read_df()
        df_train.columns = ["x0", "x1", "x2", "x3", "x4", "x5", "duration", "event"]
        cols_standardize = ["x1"]
        cols_leave = ["x0", "x2", "x3", "x4", "x5"]
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
        cols_standardize = [ 'n_prev_churns','log_days_between_subs','log_days_since_reg_init','age_at_start','log_payment_plan_days','log_plan_list_price','log_actual_amount_paid']
        cols_leave = ['is_auto_renew','is_cancel', 'city', 'gender','registered_via','strange_age', 'nan_days_since_reg_init', 'no_prev_churns']

        df_train["gender"].replace('male', 1, inplace=True)
        df_train["gender"].replace('female', 2, inplace=True)
        df_train["gender"].replace(np.NaN, 0, inplace=True)
        
        df_val["gender"].replace('male', 1, inplace=True)
        df_val["gender"].replace('female', 2, inplace=True)
        df_val["gender"].replace(np.NaN, 0, inplace=True)
        
        df_test["gender"].replace('male', 1, inplace=True)
        df_test["gender"].replace('female', 2, inplace=True)
        df_test["gender"].replace(np.NaN, 0, inplace=True)

    if not (args.dataset == 'kkobx'):
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    
    # if args.dataset=='kkbox':
    np.nan_to_num(x_train,copy=False)
    np.nan_to_num(x_val,copy=False)
    np.nan_to_num(x_test,copy=False)
        
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = x_val, y_val

    # replace zero time-to-event value with minimum
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
   
    print(f'x_train.shape: {x_train.shape}')
    # Model preparation =============================================================    
    in_features = x_train.shape[1]
    num_nodes = [args.num_nodes]* args.num_layers
    out_features = 1
    batch_norm = args.use_BN
    dropout = args.dropout
    output_bias = args.use_output_bias
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)
    net = net.to(device)
    model = CoxPH(net, tt.optim.Adam(weight_decay=args.weight_decay))

    wandb.init(project=args.dataset, 
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
    elif args.loss =='kspl':
        model.loss = DSAFTNKSPLLoss(args.an, args.sigma)
    elif args.loss =='kspl_new':
        model.loss = DSAFTNKSPLLossNew(args.an, args.sigma)

    # Training ======================================================================
    batch_size = args.batch_size
    # lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
    # best = lrfinder.get_best_lr()

    model.optimizer.set_lr(args.lr)
    
    epochs = args.epochs
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model.fit(x_train, y_train_transformed, batch_size, epochs, callbacks, verbose,
                    val_data = val_transformed, val_batch_size = batch_size)

    # Evaluation ===================================================================
    surv = get_surv(model, x_test)
    ev = EvalSurv(surv, durations_test_transformed, events_test, censor_surv='km')
    ctd = ev.concordance_td()
    time_grid = np.linspace(durations_test_transformed.min(), durations_test_transformed.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    nbll = ev.integrated_nbll(time_grid)
    val_loss = min(log.monitors['val_'].scores['loss']['score'])
    wandb.log({'val_loss':val_loss,
                'ctd':ctd,
                'ibs':ibs,
                'nbll':nbll})
    wandb.finish()


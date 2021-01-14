import numpy as np
import pandas as pd
import warnings
from lifelines import NelsonAalenFitter

def get_surv(model, x_test, timegrid = "train"):
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
    exp_residual = np.nan_to_num(np.exp(np.log(y_train) - model.predict(x_train).reshape(-1)))

    # compute exp(-theta) from test data to evaluate accelerating component
    exp_predict = np.nan_to_num(np.exp(-model.predict(x_test)).reshape(-1))
    
    # estimate cumulative baseline hazard function
    # based on training dataset
    H = NelsonAalenFitter().fit(exp_residual, event_observed = delta_train).cumulative_hazard_
    
    # extract timegrid and estimated hazards
    if timegrid == "train":
        max_time = y_train.max()
    else:
        max_time = max(H.index)
    
    if H.shape[0] * exp_predict.shape[0] >= 5 * 10e7:        
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

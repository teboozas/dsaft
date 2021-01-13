import numpy as np
import pandas as pd
import warnings
from lifelines import NelsonAalenFitter

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

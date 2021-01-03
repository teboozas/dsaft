import numpy as np
import pandas as pd
import warnings
from lifelines import KaplanMeierFitter

def get_surv(model, input):
    '''
    model: PyCox model class or compatibles
    input: covariate dataset to compute survival estimates
    '''
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    input, target = model.training_data

    y, d = target
    df = pd.DataFrame(y, d)
    df.columns = ["y"]
    df = df.sort_values(by = ["y"])
    
    y_sorted = df.values.reshape(-1)
    d_sorted = np.array(df.index, dtype = np.float32)
    
    theta_test = model.predict(x_test)

    e = (np.repeat(y_sorted.reshape(-1, 1), theta_test.shape[0], axis = 1) - np.exp(theta_test.reshape(-1)))
    
    surv_ = KaplanMeierFitter().fit(e.reshape(-1), np.array(list(d_sorted) * theta_test.shape[0])).survival_function_at_times(e.reshape(-1))
    surv = pd.DataFrame(surv_.to_numpy().reshape(-1, theta_test.shape[0]))
    surv.index = y_sorted
    surv.index.names = ["duration"]
    
    return surv


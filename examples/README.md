### KKBOX

```python

from kkbox import _DatasetKKBoxAdmin
kkbox = _DatasetKKBoxAdmin()
```

```python
try:
    df_train = kkbox.read_df()
except:
    kkbox.download_kkbox()
    df_train = kkbox.read_df()
```

```python
cols_standardize = [ 'n_prev_churns','log_days_between_subs','log_days_since_reg_init','age_at_start','log_payment_plan_days','log_plan_list_price','log_actual_amount_paid']
cols_leave = ['is_auto_renew','is_cancel', 'city', 'gender','registered_via','strange_age', 'nan_days_since_reg_init', 'no_prev_churns']
```

```python
df_train["gender"].replace('male', 1, inplace=True)
df_train["gender"].replace('female', 2, inplace=True)
df_train["gender"].replace(np.NaN, 0, inplace=True)
```

```python
np.nan_to_num(x_train,copy=False)
np.nan_to_num(x_val,copy=False)
np.nan_to_num(x_test,copy=False)
```

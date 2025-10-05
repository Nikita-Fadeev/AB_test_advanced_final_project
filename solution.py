import os
import json

import numpy as np
import pandas as pd
from scipy import stats
from flask import Flask, jsonify, request
from collections import Counter
from typing import Tuple, Dict, List

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

ALPHA = 0.05
EXP_DAYS = [36, 37, 38, 39, 40, 41, 42]

# получение данных о покупках
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])
# предобработка данных

# user_id_2_metric = df_sales.groupby('user_id')['cost'].sum().to_dict()
# get_metric = lambda user_id: user_id_2_metric.get(user_id, 0)

########################### STRATIFICATION ###########################

# define strats data
DEFAULT_STRAT = 2
get_strat = lambda x: np.where(x.isin([0,1]), x, DEFAULT_STRAT)
user_strat = (df_sales.assign(strat = lambda x: get_strat(x['channel']))
                      .groupby(['user_id', 'strat'])
                      .agg(cnt=('day', 'count'))
                      .reset_index()
                      .sort_values(['user_id', 'cnt'], ascending=False)
                      .drop_duplicates('user_id', keep='first'))
strat_weights = user_strat.value_counts('strat', normalize=True)
user_strat = (user_strat.assign(weight=lambda x: x['strat'].map(strat_weights))
                         .set_index('user_id')
                        #  
                         )
user_by_strat = user_strat['strat'].to_dict()


# define strat validation
def valid_strats(a_strats, b_strats, min_strat_size=10)->bool:
    # func check if a b groups are proper to stratification 
    a_set, b_set = set(a_strats), set(b_strats)
    # all group have same strats
    if a_set != b_set:
        return False
    # all group have more then one strat
    if len(a_set) < 2:
        return False
    # all strats have minimum size
    if any([i < min_strat_size for i in Counter(a_strats).values()]):
        return False
    if any([i < min_strat_size for i in Counter(b_strats).values()]):
        return False
    return True

# calculate strat mean and variance
def calc_strat_mean_var(df, weights)->Tuple:
    '''
    :param df: - dataframe {metric: [], strat: []}
    :param weights: - dict({'strat_1': weight, 'strat_2':'weigth' ... })

    :return tuple(mean, var): stratified mean and var
    '''
    mean = df.groupby('strat')['metric'].mean()
    strat_mean = (mean * weights).sum()
    var = df.groupby('strat')['metric'].var()
    strat_var = (var * weights).sum()
    return strat_mean, strat_var

# define strat T-test
def strat_ttest(df_a, df_b, weights, alpha)->int:
    '''
    :param df_a: - dataframe {metric: [], strat: []}
    :param df_b: - dataframe {metric: [], strat: []}
    :param weights: - dict({'strat_1': weight, 'strat_2':'weigth' ... })
    :alpha: significance level
    
    :return int: 1 - present positive effect, 0 - absent positive effect
    '''
    mean_a, var_a = calc_strat_mean_var(df_a, weights)
    mean_b, var_b = calc_strat_mean_var(df_b, weights)
    delta = mean_b - mean_a
    std = (var_a / len(df_a) + var_b / len(df_b)) ** 0.5
    left_bound = delta - stats.norm.ppf(1-alpha/2) * std
    return int(left_bound>0)


# define metric
user_id_2_metric = df_sales.loc[lambda x: x['day'].isin(EXP_DAYS)].groupby('user_id')['cost'].sum().to_dict()
get_metric = lambda user_id: user_id_2_metric.get(user_id, 0)
get_strat = lambda user_id: user_by_strat.get(user_id, DEFAULT_STRAT)

############################### CUPED ###################################

class FunctionalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, function, **params):
        self.function = function
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kwargs):
        return self.function(X, **self.params)

    def get_params(self, deep=True):
        return {"function": self.function, **self.params}

    def set_params(self, **params):
        for key, value in params.items():
            if key == "function":
                self.function = value
            else:
                self.params[key] = value
        return self

    def set_output(self, **params):
        return self
    
def functional_transformer(function):
    def builder(**params):
        return FunctionalTransformer(function, **params)
    return builder

@functional_transformer
def get_features(df:pd.DataFrame, first_exp_day:int, lookback_window:int=7, forward_window:int=7)->pd.DataFrame:
    # preperiod
    day_filter = np.arange(first_exp_day - lookback_window, first_exp_day)

    result = (df.loc[lambda x: x['day'].isin(day_filter)]
                .groupby('user_id')
                .agg(cost_mean=('cost', 'mean'),
                     cost_sum=('cost', 'sum'),
                     purchase_count=('cost', 'count')))
    
    df_channel = (df.loc[lambda x: x['day'].isin(day_filter)]
                    .assign(channel = lambda x: 'channel_' + x['channel'].astype(str))
                    .pivot_table(index='user_id', 
                                columns=['channel'], 
                                values='cost', 
                                aggfunc='sum')
                    .fillna(0))

    df_channel.columns = [''.join(col).strip() for col in df_channel.columns]
    result = result.join(df_channel)

    # post period
    day_filter = np.arange(first_exp_day, first_exp_day + forward_window)
    target = (df.loc[lambda x: x['day'].isin(day_filter)]
                .groupby('user_id')
                .agg(target=('cost', 'sum')))
    
    result = result.join(target)
    return result

def calculate_theta(y_pilot,  y_pilot_cov) -> float:
    """Вычисляем Theta.

    y_control - значения метрики во время пилота на контрольной группе
    y_pilot - значения метрики во время пилота на пилотной группе
    y_control_cov - значения ковариант на контрольной группе
    y_pilot_cov - значения ковариант на пилотной группе
    """
    if not isinstance(y_pilot, np.ndarray):
        y_pilot = np.array(y_pilot).flatten()
    if not isinstance(y_pilot_cov, np.ndarray):
        y_pilot_cov = np.array(y_pilot_cov).flatten()

    y = y_pilot
    y_cov = y_pilot_cov

    covariance = np.cov(y_cov, y)[0, 1]
    variance = y_cov.var()
    theta = covariance / variance
    return float(theta)

ISNA_FILTER = lambda df: ~df.isna().apply(any, axis=1)

# define feature processor 
feature_extractor = get_features(first_exp_day=EXP_DAYS[0], 
                                 lookback_window=20, 
                                 forward_window=len(EXP_DAYS))
# extract features
df = feature_extractor.fit_transform(df_sales.copy())
train = df.loc[lambda x: ISNA_FILTER(x)].copy()

target = 'target'
features = [c for c in df.columns if c != target] 
X, y = train[features], train[target]

# define model pipeline
transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
    ])

model = Pipeline([
    ('transform', transformer),
    ('model', LinearRegression()),
])

# train model get covarite 
model.fit(X,y)
covariate = model.predict(df[features])
user_id_2_covariate = df.assign(covariate = covariate)['covariate'].to_dict()

# calculate theta
tmp_aline = pd.DataFrame({'metric':user_id_2_metric, 'cov': user_id_2_covariate})
tmp_aline = tmp_aline.loc[ISNA_FILTER]
theta = calculate_theta(tmp_aline['metric'], tmp_aline['cov'])

# calculate metric CUPED
user_id_2_cuped = {}
MEAN = float(np.mean(list(user_id_2_covariate.values())))
for user, metric in user_id_2_metric.items():
    user_id_2_cuped[user] = metric - theta * user_id_2_covariate.get(user, MEAN)

get_metric = lambda user_id: user_id_2_cuped.get(user_id, 0)


app = Flask(__name__)


# def check_test(a, b):
#     """Проверяет гипотезу.
    
#     :param a: список id пользователей контрольной группы
#     :param b: список id пользователей экспериментальной группы
#     :return: 1 - внедряем изменение, 0 - иначе.
#     """
#     metrics_a = [get_metric(user_id) for user_id in a]
#     metrics_b = [get_metric(user_id) for user_id in b]
#     pvalue = stats.ttest_ind(metrics_a, metrics_b).pvalue
#     return int(pvalue < ALPHA)

def check_test(a, b):
    """Проверяет гипотезу.
    
    :param a: список id пользователей контрольной группы
    :param b: список id пользователей экспериментальной группы
    :return: 1 - внедряем изменение, 0 - иначе.
    """
    metrics_a = [get_metric(user_id) for user_id in a]
    metrics_b = [get_metric(user_id) for user_id in b]
    strat_a = [get_strat(user_id) for user_id in a]
    strat_b = [get_strat(user_id) for user_id in b]
    if valid_strats(strat_a, strat_b):
        df_a = pd.DataFrame({'metric': metrics_a, 'strat':strat_a})
        df_b = pd.DataFrame({'metric': metrics_b, 'strat':strat_b})
        return strat_ttest(df_a, df_b, strat_weights, alpha=ALPHA)

    pvalue = stats.ttest_ind(metrics_a, metrics_b).pvalue
    delta = np.mean(metrics_b) - np.mean(metrics_a)
    return int((pvalue < ALPHA) and (delta > 0))

@app.route('/ping')
def api_ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def api_check_test():
    test = json.loads(request.json)['test']
    has_effect = check_test(test['a'], test['b'])
    return jsonify(has_effect=has_effect)

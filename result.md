앙상블 세부 정보, 
앙상블 가중치: 0.2

{
    "class_name": "RobustScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {
        "quantile_range": [
            10,
            90
        ],
        "with_centering": true,
        "with_scaling": true
    },
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "LightGBMRegressor",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "boosting_type": "gbdt",
        "colsample_bytree": 0.4,
        "learning_rate": 0.16842263157894738,
        "max_bin": 63,
        "max_depth": 4,
        "min_data_in_leaf": 0.12802369503112465,
        "min_split_gain": 0.21052631578947367,
        "n_estimators": 200,
        "num_leaves": 15,
        "reg_alpha": 0.22499999999999998,
        "reg_lambda": 0.3,
        "subsample": 1,
        "subsample_freq": 5
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.2
{
    "class_name": "MinMaxScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {},
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "ExtraTreesRegressor",
    "module": "sklearn.ensemble",
    "param_args": [],
    "param_kwargs": {
        "bootstrap": false,
        "criterion": "squared_error",
        "max_features": 0.3,
        "min_samples_leaf": 0.005080937188890647,
        "min_samples_split": 0.0008991789964660114,
        "n_estimators": 25
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.13333333333333333

{
    "class_name": "MinMaxScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {},
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "RandomForestRegressor",
    "module": "sklearn.ensemble",
    "param_args": [],
    "param_kwargs": {
        "bootstrap": false,
        "criterion": "squared_error",
        "max_features": 0.7,
        "min_samples_leaf": 0.004196633747563344,
        "min_samples_split": 0.008991789964660124,
        "n_estimators": 25
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.06666666666666667

{
    "class_name": "StandardScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {
        "with_mean": true,
        "with_std": false
    },
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "ElasticNet",
    "module": "sklearn.linear_model",
    "param_args": [],
    "param_kwargs": {
        "alpha": 0.001,
        "l1_ratio": 0.7394736842105263
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.06666666666666667

{
    "class_name": "StandardScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {
        "with_mean": true,
        "with_std": false
    },
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "ElasticNet",
    "module": "sklearn.linear_model",
    "param_args": [],
    "param_kwargs": {
        "alpha": 0.001,
        "l1_ratio": 1
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.06666666666666667

{
    "class_name": "MinMaxScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {},
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "RandomForestRegressor",
    "module": "sklearn.ensemble",
    "param_args": [],
    "param_kwargs": {
        "bootstrap": true,
        "criterion": "squared_error",
        "max_features": "sqrt",
        "min_samples_leaf": 0.003466237459044996,
        "min_samples_split": 0.0075322213975862395,
        "n_estimators": 25
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.06666666666666667

{
    "class_name": "StandardScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {
        "with_mean": false,
        "with_std": true
    },
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "ElasticNet",
    "module": "sklearn.linear_model",
    "param_args": [],
    "param_kwargs": {
        "alpha": 0.001,
        "l1_ratio": 0.8436842105263158
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}

앙상블 가중치: 0.2

{
    "class_name": "RobustScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {
        "quantile_range": [
            10,
            90
        ],
        "with_centering": true,
        "with_scaling": false
    },
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

{
    "class_name": "ElasticNet",
    "module": "sklearn.linear_model",
    "param_args": [],
    "param_kwargs": {
        "alpha": 0.001,
        "l1_ratio": 0.21842105263157896
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
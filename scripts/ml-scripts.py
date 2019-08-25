xgb_parameters = { 
    'max_depth': (2, 12), 
    "reg_alpha": (0, 10), 
    "min_child_weight": (0,100),
    "learning_rate": (0.001, 1),
    "colsample_bytree": (0.1, 1)
}

def xgb_mape(preds, dtrain):
    labels = dtrain.get_label()
    return('mape', -np.mean(np.abs((labels - preds) / (labels+1))))

def create_pipe(max_depth, reg_alpha, min_child_weight, learning_rate, colsample_bytree):
    pipe = Pipeline([
        ('xgb', xgb.sklearn.XGBRegressor(
        objective='reg:squarederror',
        max_depth=int(max_depth), 
        reg_alpha=reg_alpha, 
        min_child_weight=min_child_weight,
        learning_rate = learning_rate,
        colsample_bytree = colsample_bytree,
        random_state=1))])
    
    return pipe

def RSS(model, x, y):
    return np.sum((y - model.predict(x))**2, axis=0)

def split_cv(model, X_train_val, y_train_val, split_type=TimeSeriesSplit(n_splits=5)):
    avg_rss = 0
    it = 0
    
    for train_index, test_index in split_type.split(X_train_val):
        X_train, X_val = X_train_val.iloc[train_index,:], X_train_val.iloc[test_index,:]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[test_index]

        model.fit(X_train, y_train)

        avg_rss += RSS(model, X_val, y_val)
        it += 1
    
    return -(avg_rss / 5)


def xgb_scoring(X, y, max_depth, reg_alpha, min_child_weight, learning_rate, colsample_bytree):
    model = create_pipe(max_depth, reg_alpha, min_child_weight, learning_rate, colsample_bytree)
    return split_cv(model, X_train_val, y_train_val)

#xgb_optimizator = BayesianOptimization(partial(xgb_scoring, X=X_train, y_val), xgb_parameters)

n_iter_search = 10

xgb_parameters = { 
    'max_depth': sp_randint(1, 15),
    'min_child_weight': sp_randint(0, 100),
    'eta': (0.1, 0.01, 1, 0.05),
    'alpha': (0.1, 0.01, 1, 0.05, 10, 15)
}


model = xgb.XGBRegressor(objective ='reg:squarederror', subsample=0.5)
xgb_m = RandomizedSearchCV(model, 
                            param_distributions=xgb_parameters,
                                   n_iter=n_iter_search, 
                         cv=TimeSeriesSplit(2))
xgb_m.fit(X_train, y_train)




def score(params):
    print("Training with params: ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train_val, label=y_train_val)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
   
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    
    score = mape(y_valid, predictions)
    # TODO: Add the importance for the selected features
    print("\tScore {0}\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'early_stopping_rounds':30,
    'objective': 'reg:squarederror',
    'seed': 1
}

best = fmin(score, space, algo=tpe.suggest, max_evals=50)

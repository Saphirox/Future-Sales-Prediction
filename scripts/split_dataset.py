## REWRITE IT INTO PIPELINES

def generate_features_from_train_set(X_train, X_test, key=['key']):
    def min_max_transformation(df, col):
        assert len(df) != 0
        assert col != ""       
        features = df[[*key, col]]\
             .groupby(key, as_index=False)\
             .agg(['max', 'min', 'std', 'mean', 'median'])
        
        features.columns = ["_".join(x) for x in features.columns.ravel()]
        return features
        
    def merge_train_test(train_df, test_df, result):
        train_df = train_df.merge(result, on=key, right_index=False, how='left')
        test_df = test_df.merge(result, on=key, right_index=False, how='left')
        return train_df, test_df

    ## Features
    
    ## Returned items
    #returned_items = X_train[X_train['item_cnt_day'] < 0].groupby('date_block_num')\
    #.sum().reset_index().rename(columns={'item_cnt_day': 'returned_items'})
    
    x_item_price = min_max_transformation(X_train, 'item_price')
   
    #months_period_of_product = X_train[['date_block_num', 'key']]\
   # .groupby(key, as_index=False).apply(lambda x: x.max() - x.min())['date_block_num']
    
    months_period_of_product.columns = ['months_period_of_product']
    
    for i, tr in enumerate([x_item_price, months_period_of_product]):
        X_train, X_test = merge_train_test(X_train, X_test, tr)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, X_test


def split_dataset(gb_df, date_col, pred_col, prediction_size = 5):
    gb_df = gb_df.copy().sort_values(by=[date_col])
    gb_df = gb_df.dropna()
    X = gb_df.drop(pred_col, axis=1)
    y = gb_df[pred_col]
    
    max_month = X[date_col].max()
    
    train_val_condition = X[date_col] < (max_month - prediction_size)
    test_condition = X[date_col] >= (max_month - prediction_size)
    
    X_train_val = X[train_val_condition]
    X_test = X[test_condition]
    y_train_val = y[train_val_condition]
    y_test = y[test_condition]
    
    return X_train_val, y_train_val, X_test, y_test

def split_and_transform(df, **kwargs):
  ## Add transformers 
    X_train_val, y_train_val, X_test, y_test = split_dataset(df, **kwargs)
    X_train_val, X_test = generate_features_from_train_set(X_train_val, X_test)    
    
    return X_train_val, y_train_val, X_test, y_test
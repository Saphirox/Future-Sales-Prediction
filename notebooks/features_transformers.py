from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from scipy.stats import randint as sp_randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# TSFRESH
from tsfresh.feature_extraction import ComprehensiveFCParameters, extract_features, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column
    
    def fit(self, train_x, train_y=None, **fit_params):
        return self
   
    def transform(self, X_train, y_train=None, **fit_params):
        X_train['month'] = X_train[self.date_column].dt.month
        X_train['year'] = X_train[self.date_column].dt.year
        X_train['year'] = X_train['year'] - X_train['year'].min()
        return X_train 
    
    def get_params(self):
        return {'date_column': self.date_column }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

def calculate_forecast_errors(y_actual, y_predicted):
    """
    Calculate MAPE of the forecast.    
    """
    e = y_test - y_predicted
    p = 100 * e / y_test
    q = np.quantile(p, (0.05, 0.95))
    return np.mean(np.abs(p[(p > q[0]) & (p < q[1])]))

def plotly_df(df, title=''):
    """Visualize all the dataframe columns as line plots."""
    common_kw = dict(x=df.index, mode='lines')
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    iplot(fig, show_link=False)


class TSFreshTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_id, column_sort, column_value, extraction_settings):
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_value = column_value
        self.extraction_settings = extraction_settings
    
    def fit(self, train_x, train_y=None, **fit_params):
        return self
   
    def transform(self, X_train, y_train=None, **fit_params):

        X_features = extract_features(
            X_train,
            column_id=self.column_id, 
            column_sort=self.column_sort, 
            column_value=self.column_value, 
            default_fc_parameters=self.extraction_settings, 
            disable_progressbar=True)
        
        impute(X_features)
        return X_features 
    
    def get_params(self):
         return {'column_id': self.column_id, 
                'column_sort': self.column_sort,
                'column_value': self.column_value,
                'extraction_settings': self.extraction_settings}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

def weight_of_evidence(goods, bads):
    return np.log(goods/bads + 0.0001) * 100

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def show_forecast(x_actual, x_predicted, y_actual, y_predicted, title, i):
    """Visualize the forecast."""
    
    def create_go(name, x, y, **kwargs):
        args = dict(name=name, x=x, y=y, mode='lines')
        args.update(kwargs)
        return go.Scatter(**args)
    
    forecast = create_go('Forecast product %i' % i, x_predicted,  y_predicted,
                         line=dict(color='rgb(31, 119, 180)'))
    actual = create_go('Actual% i' % i, x_actual, y_actual,
                       marker=dict(color="red"))
    data = [forecast, actual]
    return data

def get_default_layout(title="", xlab="", ylab=""):
    layout = go.Layout(
        title=go.layout.Title(
            text=title,
            xref='paper',
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=xlab
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=ylab
            )
        )
    )
    return layout

def show_plotly(data, title="", xlab="", ylab=""):
    plotly.offline.iplot(go.Figure(data=data, layout=get_default_layout(title,xlab,ylab)))
    
def to_sep_date(df, col):
    df['month'] = df[col].dt.month
    df['year'] = df[col].dt.year
    df['year'] = df['year'] - df['year'].min()
    return df    







import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
# from ydata_profiling import ProfileReport
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('data/data.csv')
# report = ProfileReport(df, title="Data Profiling Report", explorative=True)
# report.to_file('./output/profile_report.html')

df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

target = 'down'
y = df[target]
X = df.drop(target, axis=1)

train_ratio = 0.8
train_size = int(len(X) * train_ratio)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

class DataFrameInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, method='linear'):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.interpolate(method=self.method, axis=0).values

num_transform = Pipeline([
    ('interpolate', DataFrameInterpolator(method='linear')),
    ('scale', StandardScaler())
])

district_values = [df['District'].unique()]

ord_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder(categories=district_values))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transform, ['up', 'rnti_count', 'mcs_down', 'mcs_down_var', 'mcs_up', 'mcs_up_var', 'rb_down', 'rb_down_var', 'rb_up', 'rb_up_var']),
    ('ord', ord_transform, ['District'])
])

regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Mean Absolute Error: 7250441.422702203
# Mean Squared Error: 375017737602300.0
# R2 Score: 0.9623997666631796
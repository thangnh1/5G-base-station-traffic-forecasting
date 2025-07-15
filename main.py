import pandas as pd
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
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

# Mean Absolute Error: 6365590.877231168
# Mean Squared Error: 500843708747209.9
# R2 Score: 0.9777988150267249
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_experiment('Linear Regression')

def train_and_evaluate(df):
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric('mse', mse)
    return model, mse

def drop_and_evaluate(df):
    original_columns = df.columns
    mse_results = {}
    for col in original_columns:
        if col != 'SalePrice':
            df_dropped = df.drop(columns=[col])
            with mlflow.start_run(run_name=f"dropped-column-{col}"):
                mlflow.log_param('dropped_column', col)
                _, mse = train_and_evaluate(df_dropped)
                mse_results[col] = mse
    return mse_results

if __name__ == '__main__':
    df_uploaded = pd.read_csv('selected_1000_rows_one_hot.csv')
    mlflow.start_run()
    drop_and_evaluate(df_uploaded)
    mlflow.end_run()

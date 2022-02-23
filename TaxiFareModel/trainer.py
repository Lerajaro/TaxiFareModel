from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data, holdout
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    STUDENT_NAME = 'Lerajaro'
    
    
    def __init__(self, X, y, experiment_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.mlflow_log_param('student_name', self.STUDENT_NAME)
        self.pipeline = None
        self.experiment_name = f"[DE][Berlin][{self.STUDENT_NAME}]{experiment_name}"
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        # Add the model of your choice to the pipeline
        model = 'linear_model'
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            (model, LinearRegression())
        ])
        self.mlflow_log_param('model', model)
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.model = self.set_pipeline().fit(self.X, self.y)
        # getting the experiment_id to find it easier
        experiment_id = trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
        # end of insert
        return self.model

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.model.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()
    
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id
    
    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df_tmp = get_data()
    # clean data
    df = clean_data(df_tmp)
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = holdout(X, y)
    # iniitalize trainer
    trainer = Trainer(X_train, y_train, 'TaxiFare v1')
    # train pipeline and model
    trainer.run()
    # evaluate
    res = trainer.evaluate(X_test, y_test)
    print(f"the rmse is {res}")
  

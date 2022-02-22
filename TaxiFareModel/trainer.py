from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data, holdout
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer():
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.df = None
        self.X = None
        self.y = None

    def get_clean_df(self):
        df_tmp = get_data()
        # clean data
        df = clean_data(df_tmp)
        self.y = df.pop("fare_amount")
        self.X = df
        del df
        return self.X, self.y
        
    def traintestsplit(self):
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(self.X, self.y)
        return self.X_train, self.X_test, self.y_train, self.y_test

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
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.model = self.set_pipeline().fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.model.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return rmse

if __name__ == "__main__":
    # iniitalize trainer
    trainer = Trainer()
    # get clean data
    trainer.get_clean_df()
    # set X and y and hold out
    
    trainer.traintestsplit()
    # train pipeline and model
    trainer.run()
    # evaluate
    res = trainer.evaluate()
    print(f"the rmse is {res]")
  

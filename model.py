from config import *
from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.regularizers import l1, l2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

class HousePriceModel:
    """
    Model for House Price Prediction Competition from kaggle.com

    Functions:
        XGB: Set parameters for XGBoosting
        NN: Build Neural Network with Keras
        get_Xy: Transform pd.DataFrame to Input data and Label, given number of features
        fit: Training
    """

    def __init__(self):
        """
        :param representation_name:
            'nn' -- Neural Networks
            'xgb' -- XGBoosting
        :param num_features:
            Number of features
        """
        self.models = {}
        self.outputs = {}


    """TODO: To implement more representation, like Random Forest, etc."""
    def add_model(self, representation_name, config=None):
        # self.models store one or more models

        self.num_features = config.get('num_features', 10)

        if representation_name == 'nn':
            layers = config.get('layers', [16, 16])
            verbose = config.get('verbose', True)

            self.models['nn'] = self.NN(self.num_features, layers=layers, verbose=verbose)

        if representation_name == 'xgb':
            max_depth = config.get('max_depth', 2)
            eta = config.get('eta', 1)
            silent = config.get('silent', 1)
            nthread =config.get('nthread', 8)
            objective = config.get('objective', 'reg:linear')

            self.XGB(max_depth=max_depth,
                     eta=eta,
                     silent=silent,
                     nthread=nthread,
                     objective=objective)
        else:
            print("Error: Unavailable Representation:{}".format(representation_name))

    def XGB(self, max_depth, eta, silent, nthread, objective):
        """
        :param max_depth:
        :param eta:
        :param silent:
        :param nthread:
        :param objective:
            Defuault 'reg:linear' as linear regression
        """
        self.xgb_params = {'max_depth':max_depth,
                           'eta':eta,
                           'silent':silent,
                           'nthread':nthread,
                           'objective':objective}

    def NN(self, input_size, layers, verbose=True):
        """
        :param input_size:
            Input size, the same as number of features
        :return:
            Neural Network Model
        """
        input = Input((input_size,))
        for l in range(len(layers)):
            if l == 0:
                fc = Dense(layers[l], kernel_regularizer=l2(0.01))(input)
            else:
                fc = Dense(16, kernel_regularizer=l2(0.01))(fc)
        # dp = Dropout(0.3)(fc2)
        output = Dense(1)(fc)

        model = Model(inputs=input, outputs=output)
        if verbose:
            model.summary()

        return model

    def get_Xy(self, dataFrame, bool_train=True, method='pearson',target_feature='SalePrice'):
        """
        :param dataFrame:
            type: pandas.DataFrame()
        :param method:
            Correlation Metrics:
                'pearson': Pearson Method
                'spearman': Spearman Method
        :param target_feature:
            Default 'SalePrice', as this is the Label
        :return:
            type: numpy.ndarray()
            Input Data, Labels
        """

        if bool_train:
            self.features = select_features(dataFrame, self.num_features + 1, method=method)[1:]
            X = np.array(dataFrame[self.features])
            y = np.array(dataFrame[target_feature])

            return X, y
        else:
            X = np.array(dataFrame[self.features])

            return X

    def fit(self, representation_name, X, y, config):
        """
        :param representation_name:
            'nn' -- Neural Networks
            'xgb' -- XGBoosting
        :param X:
            Input Data
        :param y:
            Labels
        :param config:
            training configuration
        """

        epochs = config.get('epochs', 1000)
        batch_size = config.get('batch_size', 64)
        optimizer = config.get('optimizer', None)
        split = config.get('split', 0.3)
        verbose = config.get('verbose', True)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=split, random_state=2)

        if representation_name == 'nn':
            model = self.models[representation_name]

            if optimizer == None:
                optimizer = 'adam'

            model.compile(optimizer=optimizer, loss='mean_squared_error')
            self.hist = model.fit(x=self.X_train, y=self.y_train,
                                  epochs=epochs, batch_size=batch_size,
                                  validation_data=[self.X_valid, self.y_valid])
        if representation_name == 'xgb':

            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dvalid = xgb.DMatrix(self.X_valid, label=self.y_valid)

            evallist = [(dvalid, 'eval'), (dtrain, 'train')]

            self.models[representation_name] = xgb.train(params=self.xgb_params, dtrain=dtrain,
                                                         num_boost_round=epochs, evals=evallist,
                                                         verbose_eval=verbose)

    def predict(self, representation_name, X):
        """
        :param representation_name:
        :param X:
            Test Data
        :return:
            Predicted Labels
        """

        if representation_name == 'nn':
            pred = self.models['nn'].predict(X)
        if representation_name == 'xgb':
            dtest = xgb.DMatrix(X)
            pred = self.models[representation_name].predict(dtest)

        self.outputs[representation_name] = np.exp(pred).reshape((pred.shape[0],))

        return self.outputs[representation_name]

    def evaluate(self, y_true, y_pred):
        """
        :param representation_name:
        :param X:
        :return:
        """

        return mean_squared_error(y_true, y_pred)

    def fill_submission(self, y, dataFrame):
        """
        :param X:
            Predicted Labels
        :param dataFrame:
            Submission.csv to fill
        :return:
            Filled Submission.csv
        """

        dataFrame.loc[:, 'SalePrice'] = pd.Series(y)

        return dataFrame
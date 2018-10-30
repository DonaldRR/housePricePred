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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost.sklearn import XGBRegressor


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
        """
        {'model_name': {
            'name': {
                'model': model, (estimator object)
                'training_config': {
                    'num_features': num_features, (int)
                    'split': split_rate, (float)
                    Others
                    }
                }
            }
        }
        """
        self.outputs = {}


    """TODO: To implement more representation, like Random Forest, etc."""
    def add_model(self, representation_name, name=None, config=None):
        """
        :param representation_name:
            Representation name, like 'nn' or 'xgb'

        :param name:
            Name for certain model, like 'nn_1'

        :param config:
            List:[model_configuration, training_configuration]

        """

        if self.models.get(representation_name) == None:
            self.models[representation_name] = {}
        if name == None or name=='':
            name = representation_name+'_'+str(len(self.models[representation_name].keys()) + 1)
            self.models[representation_name][name] = {}

        self.models[representation_name][name]['training_config'] = config[1]

        if representation_name == 'nn':
            self.models['nn'][name]['model'] = self.NN(config[0])
        elif representation_name == 'xgb':
            self.models['xgb'][name]['model'] = self.XGB(config[0])
        else:
            print("Error: Unavailable Representation:{}".format(representation_name))

        return name

    def XGB(self, config):

        learning_rate = config.get('learning_rate', 0.1)
        n_estimators = config.get('n_estimators', 200)
        min_child_weight = config.get('min_child_weight', 3)
        booster = config.get('booster', 'gbtree')

        return XGBRegressor(learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            min_child_weight=min_child_weight,
                            booster=booster)

    def NN(self, config):

        hidden_layer_sizes = config.get('hidden_layer_sizes', (16, 16))
        activation = config.get('activation', 'relu')
        alpha = config.get('alph', 0.001)
        learning_rate = config.get('learning_rate', 'adaptive')

        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            alpha=alpha,
                            learning_rate=learning_rate)

    def get_Xy(self, dataFrame, representation_name, index=0, name=None, bool_train=True, method='pearson',target_feature='SalePrice'):
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

        if name is not None:
            model_name = name
        else:
            model_name = self.models[representation_name].keys()[index]

        num_features = self.models[representation_name][model_name]['training_config']['num_features']

        if bool_train:
            features = select_features(dataFrame, num_features + 1, method=method)[1:]
            self.models[representation_name][model_name]['features'] = features
            X = np.array(dataFrame[features])
            y = np.array(dataFrame[target_feature])

            return X, y
        else:
            features = self.models[representation_name][model_name]['features']
            X = np.array(dataFrame[features])

            return X

    def fit(self, representation_name, dataFrame, index=0, name=None):
        """
        :param representation_name:
            'nn' -- Neural Networks
            'xgb' -- XGBoosting
        :param index:
            The order of model under model_type (e.g. 1 under 'nn')
        :param name:
            The name of model under model_type (e.g. 'nn_1' under 'nn')
        """

        if name is not None:
            model_name = name
        else:
            try:
                model_name = self.models[representation_name].keys()[index]
            except:
                print("Error: Not such a model: {} index={} name={}".format(representation_name, index, name))

        model = self.models[representation_name][model_name]['model']
        training_config = self.models[representation_name][model_name]['training_config']

        if training_config.get('num_features') == None:
            training_config['num_features']= 10

        num_features = training_config.get('num_features', 10)
        split = training_config.get('split', 0.2)

        X, y = self.get_Xy(dataFrame, representation_name=representation_name, name=model_name)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split, random_state=2)

        if representation_name == 'nn':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        if representation_name == 'xgb':
            eval_set = training_config.get('eval_set', None)
            eval_metric = training_config.get('eval_metric', None)
            verbose = training_config.get('verbose', True)
            xgb_model = training_config.get('xgb_model', None)
            self.models[representation_name][model_name]['model'] = model.fit(X, y)
            # model.fit(X, y,
            #           eval_set = eval_set,
            #           eval_metric = eval_metric,
            #           verbose = verbose,
            #           xgb_model = xgb_model)

        print('\t-- Validation MSE of Model--{}: {}'.format(model_name,self.evaluate(y_valid, self.predict(representation_name, X_valid, name=model_name))))

    def predict(self, representation_name, X, index=0, name=None):
        """
        :param representation_name:
        :param dataFrame:
        :param index:
        :param name:
        :return:
        """

        if name is not None:
            model_name = name
        else:
            model_name = self.models[representation_name].keys(index)

        model = self.models[representation_name][model_name]['model']

        return np.reshape(model.predict(X), newshape=(len(X),))


    def evaluate(self, y_true, y_pred, bool_exp=False):
        """
        :param representation_name:
        :param X:
        :return:
        """

        if bool_exp:
            return mean_squared_error(y_true, y_pred)
        else:
            return mean_squared_error(np.exp(y_true), np.exp(y_pred))

    def ensemble_outputs(self, output_list, bool_exp=False):
        """
        output_list = [num_models, n_rows]
        """

        if bool_exp:
            return np.mean(output_list, axis=0)
        else:
            return np.mean(np.exp(output_list), axis=0)

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
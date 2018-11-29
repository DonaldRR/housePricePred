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

        XGB: Return XGBRegressor() given configuration
        NN: Return MLPRegressor() given configuration

        get_Xy: Transform pd.DataFrame to Input data and Label, given certain model with num_feautures set
        fit: Training
        predict: Predict outputs
        evaluate: Return MSE metrics given true labels and predicted labels
        ensemble_outputs: Compute mean of several sets of labels
        fill_submission: Fill in CSV with labels
    """

    def __init__(self):

        self.models = {}
        """
        
        ========================== IMPORTANT ===========================
        
        In this model, you need to specify certain model with 
        'representation_name' and 'index' or 'name'. As self.models can 
        store many models, we can stack those many good models to get 
        the best output. 
        
        ================================================================
        
        {'model_name': {
            'name': {
                'model': model, (estimator object)
                'num_features': num_features, (int)
                'features': features_names, (list of strings)
                'training_config': {
                    'split': split_rate, (float)
                    Others
                    }
                }
            }
        }
        """

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
            model_name = representation_name+'_'+str(len(self.models[representation_name].keys()) + 1)
        else:
            model_name = name
        self.models[representation_name][model_name] = {}

        self.models[representation_name][model_name]['training_config'] = config[1]
        if self.models[representation_name][model_name]['training_config'].get('num_features') == None:
            self.models[representation_name][model_name]['num_features']= 10

        if representation_name == 'nn':
            self.models['nn'][model_name]['model'] = self.NN(config[0])
        elif representation_name == 'xgb':
            self.models['xgb'][model_name]['model'] = self.XGB(config[0])
        elif representation_name == 'svr':
            self.models['svr'][model_name]['model'] = self.SVR(config[0])
        elif representation_name == 'randF':
            self.models['randF'][model_name]['model'] = self.RandomForest(config[0])
        elif representation_name == 'bagging':
            self.models['bagging'][model_name]['model'] = self.Bagging(config[0])
        elif representation_name == 'logistic':
            self.models['logistic'][model_name]['model'] = self.Logistic(config[0])
        elif representation_name == 'dt':
            self.models['dt'][model_name]['model'] = self.DT(config[0])
        else:
            print("Error: Unavailable Representation:{}".format(representation_name))

        return model_name

    def XGB(self, config):

        learning_rate = config.get('learning_rate', 0.1)
        n_estimators = config.get('n_estimators', 200)
        min_child_weight = config.get('min_child_weight', 3)
        booster = config.get('booster', 'gbtree')
        max_depth = config.get('max_depth', 2)
        gamma = config.get('gamma', 0.01)
        max_delta_step = config.get('max_delta_step',1)

        return XGBRegressor(learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            min_child_weight=min_child_weight,
                            booster=booster,
                            max_depth=max_depth,
                            gamma=gamma,
                            max_delta_step=max_delta_step)

    def NN(self, config):

        hidden_layer_sizes = config.get('hidden_layer_sizes', (16, 16))
        activation = config.get('activation', 'relu')
        alpha = config.get('alph', 0.001)
        learning_rate = config.get('learning_rate', 'adaptive')
        max_iter = config.get('max_iter', 200)

        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            alpha=alpha,
                            learning_rate=learning_rate,
                            max_iter=max_iter)

    def SVR(self, config):
        degree = config.get('degree',3)
        kernel = config.get('kernel','rbf')
        gamma = config.get('gamma','auto')
        coef0 = config.get('coef0',0.0)
        C = config.get('C',1.0)
        tol = config.get('tol',1e-3)

        return SVR(kernel=kernel,
                   degree=degree,
                   gamma=gamma,
                   coef0=coef0,
                   C=C,
                   tol=tol)

    def RandomForest(self, config):
        n_estimators = config.get('n_estimators', 10)
        criterion = config.get('criterion', 'mse')
        max_depth = config.get('max_depth', None)
        min_samples_split = config.get('min_samples_split', 2)
        min_samples_leaf = config.get('min_samples_leaf', 1)

        return RandomForestRegressor(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     criterion=criterion)

    def Bagging(self, config):
        base_estimator = config.get('base_estimator', None)
        n_estimators = config.get('n_estimators', 10)
        max_samples = config.get('max_samples', 1.0)
        max_features = config.get('max_features', 1.0)
        bootstrap = config.get('bootstrap', True)

        return BaggingRegressor(base_estimator=base_estimator,
                                n_estimators=n_estimators,
                                max_samples=max_samples,
                                max_features=max_features,
                                bootstrap=bootstrap)

    def Logistic(self, config):
        penalty = config.get('penalty', 'l2')
        dual = config.get('dual', False)
        tol = config.get('tol', 1e-4)
        C = config.get('C', 1.0)
        random_state = config.get('random_state', None)
        solver = config.get('solver', 'liblinear')
        max_iter = config.get('max_iter', 100)

        return LogisticRegression(penalty=penalty,
                                  dual=dual,
                                  tol=tol,
                                  C=C,
                                  random_state=random_state,
                                  solver=solver,
                                  max_iter=max_iter)

    def DT(self, config):
        criterion = config.get('criterion', 'mse')
        splitter = config.get('splitter', 'best')
        max_depth = config.get('max_depth', None)
        min_samples_split = config.get('min_samples_split', 2)
        min_samples_leaf = config.get('min_samples_leaf', 1)
        min_weight_fraction_leaf = config.get('min_weight_fraction_leaf', 0.)
        min_impurity_decrease = config.get('min_impurity_decrease', 1e-7)

        return DecisionTreeRegressor(criterion=criterion,
                                     splitter=splitter,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                                     min_impurity_decrease=min_impurity_decrease)

    def get_Xy(self, dataFrame, representation_name, index=0, name=None, bool_train=True, method='pearson',target_feature='SalePrice'):
        """
        :param dataFrame:
        :param representation_name:
        :param index:
        :param name:
            Model Name
        :param bool_train:
            The difference between training and test dataFrame() is whether it has 'SalePrice' column or not
        :param method:
            'pearson' or 'spearman'
        :param target_feature:
        :return:
            X and y or X, numpy.ndarray()
        """

        if name is not None:
            model_name = name
        else:
            model_name = self.models[representation_name].keys()[index]

        num_features = self.models[representation_name][model_name]['num_features']

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

        split = training_config.get('split', 0.2)

        X, y = self.get_Xy(dataFrame, representation_name=representation_name, name=model_name)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split, random_state=2)

        if representation_name == 'nn':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        elif representation_name == 'svr':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        elif representation_name == 'randF':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        elif representation_name == 'bagging':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        elif representation_name == 'logistic':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        elif representation_name == 'dt':
            self.models[representation_name][model_name]['model'] = model.fit(X_train, y_train)
        elif representation_name == 'xgb':
            eval_set = training_config.get('eval_set', None)
            eval_metric = training_config.get('eval_metric', None)
            verbose = training_config.get('verbose', True)
            xgb_model = training_config.get('xgb_model', None)
            self.models[representation_name][model_name]['model'] = model.fit(X, y,
                                                                              eval_set = eval_set,
                                                                              eval_metric = eval_metric,
                                                                              verbose = verbose,
                                                                              xgb_model = xgb_model)


        print('\t-- Validation MSE of Model--{}: {}'.format(model_name,self.evaluate(y_valid, self.predict(representation_name, X_valid, name=model_name))))

    def predict(self, representation_name, X, index=0, name=None):

        if name is not None:
            model_name = name
        else:
            model_name = self.models[representation_name].keys(index)

        model = self.models[representation_name][model_name]['model']

        return np.reshape(model.predict(X), newshape=(len(X),))


    def evaluate(self, y_true, y_pred):

        return mean_squared_error(y_true, y_pred)

    def ensemble_outputs(self, output_list):
        """
        :param output_list:
            [num_models, n_rows]
        """

        return np.mean(output_list, axis=0)

    def fill_submission(self, y, dataFrame):

        dataFrame.loc[:, 'SalePrice'] = pd.Series(y)

        return dataFrame
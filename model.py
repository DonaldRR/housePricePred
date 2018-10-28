from config import *
from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.regularizers import l1, l2
import xgboost as xgb

class HousePriceModel:
    """
    Model for House Price Prediction Competition from kaggle.com

    Functions:
        XGB: Set parameters for XGBoosting
        NN: Build Neural Network with Keras
        get_Xy: Transform pd.DataFrame to Input data and Label, given number of features
        fit: Training
    """

    def __init__(self, representation_name, num_features):
        """
        TODO: To implement more representation, like Random Forest, etc.
        :param representation_name:
            'nn' -- Neural Networks
            'xgb' -- XGBoosting
        :param num_features:
            Number of features
        """
        self.representation = representation_name
        self.num_features = num_features

        # self.models store one or more models
        self.models = {}
        if self.representation == 'nn':
            self.models['nn'] = self.NN(num_features)
        if self.representation == 'xgb':
            self.models['xgb'] = self.XGB()
        else:
            print("Error: Unavailable Representation:{}".format(representation_name))

    def XGB(self, max_depth=2, eta=1, silent=1, nthread=4, objective='reg:linear'):
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

    def NN(self, input_size):
        """
        :param input_size:
            Input size, the same as number of features
        :return:
            Neural Network Model
        """
        input = Input((input_size,))
        fc1 = Dense(16, kernel_regularizer=l2(0.01))(input)
        fc2 = Dense(16, kernel_regularizer=l2(0.01))(fc1)
        fc3 = Dense(16, kernel_regularizer=l2(0.01))(fc2)
        # dp = Dropout(0.3)(fc2)
        output = Dense(1)(fc3)

        model = Model(inputs=input, outputs=output)
        model.summary()

        return model

    def get_Xy(self, dataFrame, method='pearson',target_feature='SalePrice'):
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

        features = select_features(dataFrame, self.num_features + 1, method=method)

        X = np.array(dataFrame[features[1:]])
        y = np.array(dataFrame[target_feature])

        return X, y

    def fit(self, representation_name, X, y, epochs=1000, batch_size=64, optimizer=None, split=0.3):
        """
        :param representation_name:
            'nn' -- Neural Networks
            'xgb' -- XGBoosting
        :param X:
            Input Data
        :param y:
            Labels
        :param epochs:
            Training epochs
        :param batch_size:
            Training Batch size, for neural network
        :param optimizer:
            type: any object or string, specifically
        :param split:
            Ratio of validation set to overall data set
        """

        model = self.models[representation_name]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=split, random_state=2)

        if representation_name == 'nn':

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

            xgb.train(self.xgb_params, dtrain, epochs, evallist)
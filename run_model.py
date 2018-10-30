from config import *
from utils import *
from model import HousePriceModel


"""House Price Predicting

This Code Implement Ensemble Learning Capability, which means it can train on several (different) models, 
and integrate all the outputs as final outpus.

Arguments are names of models, like xgb, nn. Configurations for models are set Default.

"""

if __name__ == '__main__':
    train_data = pd.read_csv(PREPROCESSED_TRAINING_DATA_PATH)
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
    test_data = pd.read_csv(PREPROCESSED_TEST_DATA_PATH)
    test_data = test_data.drop(['Unnamed: 0'], axis=1)

    args = sys.argv[1:]

    CLFs = HousePriceModel()

    ensemble_models = []
    """
    ensemble_models contains a set of models, to perform ensemble learning on several models.

    It contains a List of models, for each #element:
        [0. representation name,
        1. model name,
        2. [model config(dict), training config (dict)]

        Sample cofig: (No specification is allowed, which means using default values)
        
            !!! Check those PARAMETERS on Sklearn Package !!!
        
            I. model config
                a) xgb
                    {
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'min_child_weight': min_child_weight,
                    'booster': booster
                    }
                b) nn
                    {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': activation,
                    'alpha': alpha,
                    'learning_rate': learning_rate
                    }

            II. training config (dict)
                a) xgb
                    {
                    'num_features': num_features,
                    'split': split,
                    'eval_set': eval_set,
                    'eval_metric': eval_metric,
                    'verbose': verbose,
                    'xgb_model': xgb_model
                    }
                b) nn
                    {
                    'num_features': num_features,
                    'split': split,
                    }
    """

    if len(args) > 0:
        for arg in args:
            ensemble_models.append([arg, {}, {}])
    else:
        # Default conbination of models
        ensemble_models = [
            ['xgb',
             '',
             [{}, {}]
             ],
            ['xgb',
             '',
             [{}, {}]
             ],
            ['xgb',
             '',
             [{},{}]
             ]
        ]

    m = train_data.shape[0]
    y_train = np.reshape(train_data['SalePrice'], newshape=(m,))
    train_output_list = []
    test_output_list = []

    print('== Starting Training ...')
    for m in range(len(ensemble_models)):

        cur_model = ensemble_models[m]

        rps_name = cur_model[0]
        model_name = cur_model[1]
        config = cur_model[2]

        model_name = CLFs.add_model(representation_name=rps_name, config=config)
        CLFs.fit(representation_name=rps_name, dataFrame=train_data, name=model_name)

        X_train, y_train_ = CLFs.get_Xy(train_data, representation_name=rps_name, name=model_name)
        X_test = CLFs.get_Xy(test_data, representation_name=rps_name, name=model_name, bool_train=False)

        train_output = CLFs.predict(representation_name=rps_name, X=X_train, name=model_name)
        test_output = CLFs.predict(representation_name=rps_name, X=X_test, name=model_name)

        train_output_list.append(train_output)
        test_output_list.append(test_output)

        print("\t-- Overall MSE of Model-{}: {}".format(model_name, CLFs.evaluate(y_train, train_output)))

    print('== Ensemble mse:{}'.format(CLFs.evaluate(np.exp(y_train), CLFs.ensemble_outputs(train_output_list), bool_exp=True)))

    # Integrate all outputs from several models
    final_output = CLFs.ensemble_outputs(test_output_list)

    print('== Fill in submission ...')
    # Fill submission.csv
    submission = pd.read_csv(SUBMISSION_PATH)
    CLFs.fill_submission(final_output, submission)

    print("== Process Successed!")
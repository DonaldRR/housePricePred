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

    clf = HousePriceModel()

    ensemble_models = []
    """
    ensemble_models contains a set of models, to perform ensemble learning on several models.

    It contains a List of models, for each #element:
        [0. representation name,
        1. model config(dict),
        2. training config (dict)]

        Sample cofig: (No specification is allowed, which means using default values)

            I. model config
                a) xgb
                    {'max_depth':int,
                    'eta':int,
                    'silent':int,
                    'nthread':int,
                    'objective':string, default 'reg:linear'
                    }
                b) nn
                    {'layers':[16,16], number of units for each fully connected layer
                    'verbose':Boolean, 
                    }

            II. training config (dict)
                {'epochs':int,
                'batch_size':int,
                'optimizer': str or object
                'split':float
                }
    """

    if len(args) > 0:
        for arg in args:
            ensemble_models.append([arg, {}, {}])
    else:
        # Default conbination of models
        ensemble_models = [
            ['xgb',
             {},
             {}],
            ['xgb',
             {},
             {}],
            ['nn',
             {},
             {}]
        ]


    output_list = []
    print('Starting Training ...')
    for m in range(len(ensemble_models)):

        cur_model = ensemble_models[m]
        rps_name = cur_model[0]
        model_config = cur_model[1]
        training_config = cur_model[2]

        clf.add_model(representation_name=rps_name, config=model_config)    # add model
        X, y = clf.get_Xy(train_data, True, method='spearman')  # get training input data from dataFrame
        clf.fit(representation_name=rps_name, X=X, y=y, config=training_config) # training data
        X_test = clf.get_Xy(test_data, False)   # get test input data
        output_list.append(clf.predict(representation_name=rps_name, X=X_test)) # ger predicted output

    # Integrate all outputs from several models
    final_output = ensemble_outputs(output_list=output_list)

    print('Fill in submission ...')
    # Fill submission.csv
    submission = pd.read_csv(SUBMISSION_PATH)
    clf.fill_submission(final_output, submission)

    print("Process Successed!")
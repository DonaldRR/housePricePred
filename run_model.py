from config import *
from utils import *
from model import HousePriceModel
"""
Arguments are:
    string: Representation name,
    int:    Number of features,
    int:    Training epochs
"""


if __name__ == '__main__':
    train_data = pd.read_csv(PREPROCESSED_TRAINING_DATA_PATH)
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

    args = sys.argv[1:]

    try:
        represtation_name = args[0]
    except:
        represtation_name = 'xgb'
    try:
        num_features = args[1]
    except:
        num_features = 15
    try:
        epochs = args[2]
    except:
        epochs = 3000

    print("===============Current Model Configuration=================")
    print("represtation_name:{}".format(represtation_name))
    print("num_features:{}".format(num_features))
    print("epochs:{}".format(epochs))
    print("===========================================================")

    clf = HousePriceModel(representation_name=represtation_name, num_features=15)
    X, y = clf.get_Xy(train_data, method='spearman')
    print("Start Training ... ")
    clf.fit(represtation_name, X, y, 1000)
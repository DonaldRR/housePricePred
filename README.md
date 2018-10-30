# housePricePred

Code for House Prices: Advanced Regression Techniques(https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Environment & Dependencies:

* Python: 3.6
* numpy(1.14.5)
* pandas(0.23.4)
* scipy(1.1.0)
* matplotlib(2.2.3)
* tensorflow(1.10.1)
* xgboost(0.80)
* seaborn
* sklearn(>=0.19.2)

(Versions are not limited as stated above, they are just my laptop's configuration. Later versions are likely to work.)

## Install
Install project in your local computer. (Install `git` before, with `pip install git`)
```Bash
git clone https://github.com/DonaldRR/housePricePred.git
```


## Datasets
You need to download Datasets from [kaggle's House Prices:Advanced Regression website](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data),
or using command line (if this works...)
```Bash
kaggle competitions download -c house-prices-advanced-regression-techniques
```

By default, you need to make two directories(`datasets` and `processed_data`) under your project directory `CurrentProjectDir` to store data, like
```
-->CurrentProjectDir
  -->datasets
  -->processed_data
```

Of course, you can change directories of datasets in `config.py`

## Run Scripts

### Preprocess
First, process your original CSVs.
```Bash
python preprocess.py
```
Processed files will occur under `processed_data` directory

### Run Models
Run specific model to train the data.
Arguments are available,

`model_type`,  'nn' or 'xgb' available

```Bash
python run_model.py [model_type [model_type [model_type [...]]]
```

## More

`TODOs` in `model.py` are what you need to complete.

from config import *
from utils import *

# =================================
# ====== Read Original Data =======
# =================================

print("== 1. Reading Data ...")
df_train = pd.read_csv(TRAINING_DATA_PATH)
df_test = pd.read_csv(TEST_DATA_PATH)
df_allX = pd.concat([df_train.loc[:, 'MSSubClass':'SaleCondition'],
                     df_test.loc[:,'MSSubClass':'SaleCondition']])
df_allX = df_allX.reset_index(drop=True)

# Features of different types
feats_numeric = df_allX.dtypes[df_allX.dtypes != "object"].index.values
feats_object = df_allX.dtypes[df_allX.dtypes == "object"].index.values

feats_numeric_discrete = ['MSSubClass', 'OverallQual', 'OverallCond']
feats_numeric_discrete += ['TotRmsAbvGrd', 'KitchenAbvGr', 'BedroomAbvGr', 'GarageCars', 'Fireplaces']
feats_numeric_discrete += ['FullBath', 'HalfBath','BsmtHalfBath','BsmtFullBath']
feats_numeric_discrete += ['MoSold', 'YrSold']

# Continue-valued and Discrete-valued attributes
feats_continu = feats_numeric.copy()
feats_discrete = feats_object.copy()

for f in feats_numeric_discrete:
    feats_continu = np.delete(feats_continu, np.where(feats_continu == f))
    feats_discrete = np.append(feats_discrete, f)


# =================================
# ======= Preprocess Data =========
# =================================

# Drop Unuseful Features
print("== 2. Drop Usefulless Features ...")
feats_del = ['YrSold', 'MoSold']
df_allX.drop(feats_del, axis=1, inplace=True)

for f in feats_del:
    feats_numeric = np.delete(feats_numeric, np.where(feats_numeric == f))
    feats_object = np.delete(feats_object, np.where(feats_object == f))
    feats_continu = np.delete(feats_continu, np.where(feats_continu == f))
    feats_discrete = np.delete(feats_discrete, np.where(feats_discrete == f))

print("== 3. Drop Outliers ... ")
# Drop Outliers
ids = []

ids.extend(outlierId(df_train, 'LotFrontage', 2))
ids.extend(outlierId(df_train, 'LotArea', 4))
ids.extend(outlierId(df_train, 'BsmtFinSF1', 1))
ids.extend(outlierId(df_train, 'BsmtFinSF2', 1))
ids.extend(outlierId(df_train, '1stFlrSF', 1))
ids.extend(outlierId(df_train, 'GrLivArea', 2))
ids.extend(outlierId(df_train, 'TotalBsmtSF', 1))

for id in np.unique(ids):
    df_train = df_train.drop(df_train[df_train.Id==id].index)
    df_allX = df_allX.drop(df_allX[df_allX.index==(id-1)].index)


print("== 4. Replace NULL values ...")
# Replace all NULL values for Numeric Data
df_allX = df_allX.fillna(df_allX.mean())

for c in feats_object:
    transNaNtoNA(df_allX, c)

# df_allX[feats_numeric] = df_allX[feats_numeric].apply(lambda x: (x-x.mean())/(x.std()))

dfc = df_train.copy()

print("== 5. Encoding Categorical Features ... ")
for fb in feats_object:
    # print("\r\n-----\r\n",fb,end=':')
    transNaNtoNA(dfc, fb)
    for attr_v, score in encode(dfc, fb).items():
        # print(attr_v, score, end='\t')
        df_allX.loc[df_allX[fb] == attr_v, fb] = score

stillNA = NARatio(df_allX, df_allX.columns.values)

dftemp = df_allX.copy()
for sn in stillNA.keys():
    dftemp  = transNAtoNumber(dftemp,sn)
    df_allX = transNAtoNumber(df_allX,sn,dftemp[sn].mean())


# =================================
# ========== Save Data ============
# =================================
print("== 6. Save CSVs ...")
num_train = df_train.shape[0]
tmp_df_train = df_allX[:num_train].copy()
tmp_df_train.loc[:, 'SalePrice'] = df_train['SalePrice'].values
tmp_df_train.to_csv(PREPROCESSED_TRAINING_DATA_PATH)
df_allX[num_train:].to_csv(PREPROCESSED_TEST_DATA_PATH)
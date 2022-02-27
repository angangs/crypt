from gluonts.model import deepar
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.mx.distribution import StudentTOutput

# #### Download Alphavantage data and format
import requests
import pandas as pd

symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'SOL']
exchange = 'USD'
start_date = '2020-08-20'
api_key = 'VYEPC9HMYKP99XSY'
df_dict = {}
for symbol in symbols:
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    try:
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    except Exception as e:
        print(e)
    df = df[['1a. open (USD)', '2a. high (USD)', '3a. low (USD)', '4a. close (USD)', '5. volume']]
    df = df.rename(
        columns={'1a. open (USD)': 'open', '2a. high (USD)': 'high', '3a. low (USD)': 'low', '4a. close (USD)': 'close',
                 '5. volume': 'volume'})
    df = df.astype(float)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ds'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index(['ds'], inplace=True)
    df.sort_index(inplace=True)
    df_dict[symbol] = df.loc[start_date:]

for symbol in symbols:
    print(symbol, df_dict[symbol].index[0], df_dict[symbol].index[-1])

# #### DeepAR model
categories_df = pd.DataFrame({"Group": [1, 1, 2, 2, 2]}, index=df_dict.keys())

model_params = {
    'freq': '1D',
    'prediction_length': 15,
    'num_layers': 4,
    'num_cells': 64,
    'cell_type': "lstm",
    'distr_output': StudentTOutput(),
    'dropout_rate': 0.00,
    'context_length': 15,
    'scaling': True,
    'use_feat_dynamic_real': False,
    'use_feat_static_cat': True,
    'cardinality': list(categories_df.nunique(axis=0).values)
}

trainer_params = {
    'epochs': 200,
    'learning_rate': 0.01,
    'patience': 5,
    'num_batches_per_epoch': 50
}

trainer = Trainer(**trainer_params)
estimator = deepar.DeepAREstimator(**model_params, trainer=trainer)

# #### Training
df_train_dict = {}
for symbol, df in df_dict.items():
    df_train_dict[symbol] = df.iloc[:-model_params['prediction_length']]

data = ListDataset(
    [
        {
            "start": df_train.index[0],
            "target": df_train.close,
            "feat_dynamic_real": df_train.drop(['close'], axis=1).T,
            "feat_static_cat": categories_df.loc[symbol].values
        }
        for symbol, df_train in df_train_dict.items()
    ],
    freq="1D"
)

predictor = estimator.train(training_data=data)

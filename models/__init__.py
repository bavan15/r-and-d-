from .random_forest import RandomForest
from .sarimax import Sarimax
from .orbit import Orbit
from .LSTM import MyLSTM
from .GRU import MyGRU
from .arima import MyARIMA
from .prophet import MyProphet
from .xgboost import MyXGboost
from .neural_prophet import Neural_Prophet


MODELS = {'random_forest': RandomForest,
          'xgboost': MyXGboost,
          }

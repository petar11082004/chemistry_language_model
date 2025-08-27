import pandas as pd
import torch, numpy as np
from neural_net import T1Config, T1Regressor, T1Dataset
from t1_data_processor import T1DataProcessor

train_df = pd.read_parquet("t1_pairs_training_20250815_170401.parquet")
val_df = pd.read_parquet("t1_pairs_testing_20250812_140206.parquet")
print(train_df['t1'].describe(percentiles=[.01,.5,.99]))
proc = T1DataProcessor()
proc.fit(train_df)
train_df = proc.transform(train_df)
print(train_df['t1'].describe(percentiles=[.01,.5,.99]))
'''
# train
cfg = T1Config()
reg = T1Regressor(cfg)
reg.fit(train_df = train_df, val_df = val_df, epochs=200, batch_size=64)

# save
reg.save("t1_model.pt", "t1_processor.pkl")
'''

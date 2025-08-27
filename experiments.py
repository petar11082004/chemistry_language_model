import pandas as pd
import torch, numpy as np
from neural_net import T1Config, T1Regressor, T1Dataset

train_df = pd.read_parquet("t1_pairs_training_20250812_140206.parquet")
val_df = pd.read_parquet("t1_pairs_testing_20250812_140206.parquet")

# train
cfg = T1Config()
reg = T1Regressor(cfg)
reg.fit(train_df = train_df, val_df = val_df, epochs=200, batch_size=64)

# save
reg.save("t1_model.pt", "t1_processor.pkl")


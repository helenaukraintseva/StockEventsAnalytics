import pandas as pd

df = pd.read_csv("re_0a1_i1m_w20_s1_p1.csv")[:10000]
df.to_csv("train_price.csv")
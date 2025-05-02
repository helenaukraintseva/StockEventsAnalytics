import pandas

df = pandas.read_csv("crypto_data/1000BONKUSDC_1m.csv")
print(df.columns)
print(df.columns.tolist())
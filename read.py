import pandas as pd
col_names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
csv = pd.read_csv("C:/Users/karti/Documents/EURUSD_Candlestick_1_M_BID_01.01.2019-31.12.2021.csv", names=col_names)
csv.drop('Volume', axis=1, inplace=True)
csv.drop(0 , axis=0, inplace=True)
print(csv)

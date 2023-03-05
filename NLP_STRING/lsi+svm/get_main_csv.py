import pandas as pd

f1 = pd.read_csv('/Users/shilin/Desktop/samples/data.csv')
f2 = pd.read_csv('/Users/shilin/Desktop/samples/data1.csv')
file = [f1, f2]
train = pd.concat(file)
train.to_csv("main_data" + ".csv", index=0, sep=',')
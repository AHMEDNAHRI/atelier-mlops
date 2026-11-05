import pandas as pd

df = pd.read_csv("data/raw/iris.csv")
df.sample(30, random_state=1).to_csv("data/raw/iris_sample.csv", index=False)
print("iris_sample.csv genere")

import pandas as pd

df = pd.read_csv("Dengue-Training-14000-Balanced.csv")
dups = df[df.duplicated()]
clean = df.drop_duplicates()
print(dups)
print(clean)


import pandas as pd, numpy as np
DATA="/home/rtikes/Blockchain-Distributed-IDS/data/CICIDS2017_full.csv"
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
rng = np.random.RandomState(128)
ben = df[df.Label=="BENIGN"].sample(n=32, random_state=128)
att = df[df.Label!="BENIGN"].sample(n=32, random_state=128)
canary = pd.concat([ben, att]).sample(frac=1.0, random_state=128)
canary.to_csv("fedavg_xai/canary_64.csv", index=False)
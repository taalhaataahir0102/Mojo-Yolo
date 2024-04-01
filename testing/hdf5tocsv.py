import pandas as pd
import sys
fpath = sys.argv[1]
if len(sys.argv)>2:
    key = sys.argv[2]
    df = pd.read_hdf(fpath, key=key)
else:
    df = pd.read_hdf(fpath)

df.to_csv(sys.stdout, index=False)

# python3 hdf5tocsv.py model_weights.h5 > data.csv
this is code that implement the transformer for playlist recommendation.
different from traditional transformer is that at each step, it optimize a set rather a single item.

for preparing data,  please first run 
python dataprepare.py
python datatoken.py count playlist_20181023_parse.csv
python datatoken.py token playlist_20181023_parse.csv

for training:
python 
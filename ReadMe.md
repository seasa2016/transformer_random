this is code that implement the transformer for playlist recommendation.
different from traditional transformer is that at each step, it optimize a set rather a single item.

for preparing data,  please first run 
python generative_raw.py
python datatoken.py count raw.csv
python datatoken.py token raw.csv
python change.py subword.source subword.source

for training:
python train.py -save_model {path to save} -show

for continue training:
python train.py -save_model {path to save} -train_from {path to checkpoint} -show

for testing:
python test.py -src {path to input} -test_from {path to checkpoint} -verbose -show -output {path to output} -data ./token_data/
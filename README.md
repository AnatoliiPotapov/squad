### This is not yet complete!

Experiments on SQuAD dataset with Keras.
The purpose is to make clean and efficient architecture which is easily reproducible and achieve close to state of the art results.

#### This work is highly inspired by DRQA [here](https://arxiv.org/abs/1704.00051) and FastQA [there](https://arxiv.org/abs/1703.04816)

## Results 

 

## Instructions

1. We need to parse and split the data
```sh
    python parse_data.py data/train-v1.1.json --train_ratio 0.9 --outfile data/train_parsed.json --outfile_valid data/valid_parsed.json
    python parse_data.py data/train-v1.1.json --outfile data/train_parsed.json
```

2. Preprocess the data
```sh
    python preprocessing.py data/train_parsed.json --outfile data/train_data.pkl
    python preprocessing.py data/valid_parsed.json --outfile data/valid_data.pkl
    python preprocessing.py data/dev_parsed.json --outfile data/dev_data.pkl
```

3. Train the model
```sh
    python train.py --hdim 40 --batch_size 70 --nb_epochs 50 --optimizer adam --dropout 0.2
```

4. Predict on dev/test set samples
```sh
    python predict.py model/your-model prediction.json
```





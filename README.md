##This repository contains EMO-DETECTOR.


-Download the [GloVe](http://nlp.stanford.edu/data/glove.42B.300d.zip) dataset.

-Generate embeddings using ```python3 embeddings.py -d ./data/glove.42B.300d.txt --npy_output ./data/newembeddings.npy --dict_output ./data/newvocab.pckl --dict_whitelist ./data/sorted.vocab```

-Data sorted folders are as follows: /data = training dataset | /test = test dataset

# Baselines
All code for training and reproducing results for baseline models.

To generate variable-length (local context) embeddings from fixed-length (local context) embeddings using the `AbbrRep` class, run the following:

```
python localglobalembed.py \
    -dataset="<filename>"    
    -outputfile="<outfilename>"
    -variable_local     # variable length local embedding
    -g                  # include global context
    -window=5           # use a local context of 5
```

To train the CNN on a variable-length (local context) embedding (the CNN is designed to be trained on this kind of data), run the following:

```
python train_cnn.py \
    -dataset="<filename>" \
    -num_epochs=10 \
    -ns=500
```


where `<filename>` is the name of the pickled list of `AbbrRep` objects, e.g. `<filename>=ivf_mimic_casi_w5_ns1000_g_20190408.pickle`, and typically we have `<outfilename>=var_<filename>`

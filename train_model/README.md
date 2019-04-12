### Creating Reverse-Substitution Dataset from Corpus
```
python3 create_rs_dataset.py \
    -inputfile=<path_to_abbr_file> \
    -find_a \
    -window=20 \
    -corpus=<path_to_raw_txt_file>
```

### Creating Word Embedding Representations from Dataset Samples
Each sample has AbbrRep format (whenever using datasets created by this script, include line "from localglobalembed import AbbrRep" in new script).
Uses data in ./sentences_from_mimic/, and will output data in ./embedding_files/
```
python localglobalembed.py \
    -abbr=ivf \ 
    -mimic_rs=sentences_from_mimic/ivf_mimic_rs.txt
    -mimic_abbr=sentences_from_mimic/ivf_mimic_abbr.txt
    -casi_abbr=sentences_from_casi/ivf_casi_abbr.txt 
    -window=5 
    -g 
    -variable_local 
    -outputfile=embedding_files/variable_ivf_dataset_rs_sim_abbr_casi.pickle
```


### Train Feed-Forward NN
`python3 feed_forward_nn.py`
Train feed-forward NN with following datasets (1) Reverse-substitution (original), (2) Reverse-substitution incl. samples from neighbouring words, (3) Step (1) + labelling unlabelled data with GDA, (4) Step (2) + labelling unlabelled data with GDA, (5) Step (1) + labelling unlabelled data with KNN, (6) Step (2) + labelling unlabelled data with KNN. Note this assumes pickle file contains: mimic_rs, mimic_rs_sim, casi_abbr, mimic_abbr
    Sample input format: ./embedding_files/
```
python feed_forward_nn.py \
    -dataset="./embedding_files/ivf_dataset_rs_sim_abbr_casi.pickle"
```

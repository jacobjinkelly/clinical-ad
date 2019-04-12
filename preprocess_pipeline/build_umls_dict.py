import os
import pickle
import re

mimic_file_src = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "mimic_dict.pickle")
pickle_in = open(mimic_file_src, 'rb')
mimic_dict = pickle.load(pickle_in)
pickle_in.close()

umls_file_src = "umls_cuiDescriptions_20190505.pickle"
pickle_in = open(umls_file_src, 'rb')
umls_dict = pickle.load(pickle_in)
pickle_in.close()

umls_file_dest = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "umls_mimic_union_dict.txt")
f = open(umls_file_dest, 'w')

umls_terms = umls_dict['name2id']

for key in umls_terms:

    present = True
    try:
        key = re.sub(r"'", "", key)
        tokens = key.split()
    except:
        print(key, umls_terms[key])
        continue
    for token in tokens:
        if token not in mimic_dict:
            present = False
    if present:
        f.write(key + '\n')

f.close()

import pickle

pickle_in = open("umls_cuiDescriptions_20190305.pickle", 'rb')
dct = pickle.load(pickle_in)
pickle_in.close()

dest_file = open("all_umls_terms.txt", 'w')
for key in dct[]:
    dest_file.write(key + '\n')
dest_file.close()
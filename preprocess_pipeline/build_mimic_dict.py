import os
import pickle

mimic_dict = {}
mimic_file_src = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "mimicnotes_cleaned.txt")
f = open(mimic_file_src, 'r')
for line in f:
    line = line[:-1]
    tokens = line.split()
    for t in tokens:
        try:
            mimic_dict[t] += 1
        except KeyError:
            mimic_dict[t] = 1
f.close()

file_dest = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "mimic_dict.pickle")
pickle_out = open(file_dest, 'wb')
pickle.dump(mimic_dict, pickle_out)
pickle_out.close()

print("Length of mimic dict is: " + str(len(mimic_dict)))
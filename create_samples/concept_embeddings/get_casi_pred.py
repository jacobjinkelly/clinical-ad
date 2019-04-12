from ast import literal_eval

casi_correct_terms = set()
f = open("master_tf_model_500S_10w_e49_20190408_50abbrs_output.txt", 'r')
correct_terms = set()
counter = 0
g = open("output.txt", 'w')
flag = True
for line in f:
    if flag:
        g.write(line)
    if "from casi" in line:
        flag = False
    if "from mimic" in line:
        flag = True
    counter += 1
g.close()

f = open("indiv_tf_16w_500S_e49_20190408_50abbrs_output.txt", 'r')
correct_terms = set()
counter = 0
g = open("output.txt", 'a')
flag = True
for line in f:
    if flag:
        g.write(line)
    if "from casi" in line:
        flag = False
    if "from mimic" in line:
        flag = True
    counter += 1
g.close()

f = open("indiv_tf_16w_1000S_e49_20190408_50abbrs_output.txt", 'r')
correct_terms = set()
counter = 0
g = open("output.txt", 'a')
flag = True
for line in f:
    if flag:
        g.write(line)
    if "from casi" in line:
        flag = False
    if "from mimic" in line:
        flag = True
    counter += 1
g.close()

f = open("indiv_tf_10w_1000S_e49_20190408_50abbrs_output.txt", 'r')
correct_terms = set()
counter = 0
g = open("output.txt", 'a')
flag = True
for line in f:
    if flag:
        g.write(line)
    if "from casi" in line:
        flag = False
    if "from mimic" in line:
        flag = True
    counter += 1
g.close()

f = open("indiv_tf_10w_500S_e49_20190408_50abbrs_output.txt", 'r')
correct_terms = set()
counter = 0
g = open("output.txt", 'a')
flag = True
for line in f:
    if flag:
        g.write(line)
    if "from casi" in line:
        flag = False
    if "from mimic" in line:
        flag = True
    counter += 1
g.close()

f = open("output.txt", 'r')
for line in f:
    if line[0] != "{":
        continue
    content = line[:-1]
    terms = content.split("} {")
    lhs = terms[0][1:]
    rhs = terms[1][:-1]

    if lhs == rhs:
        if "in vitro" in lhs:
            print("HI")
        lhs = lhs.replace("'", "")
        lhs_terms = lhs.split(",")
        for term in lhs_terms:
            casi_correct_terms.add(' '.join(term.split()))

g = open("../../mimicnotes_cleaned.txt", 'r')
word_counter = {}
for term in casi_correct_terms:
    word_counter[term] = 0


for line in g:
    for term in casi_correct_terms:
        new_term = " " + term + " "
        if new_term in line:
            word_counter[term] += 1

print(word_counter)
import pickle
t = open("mimic_wc.pickle", 'wb')
pickle.dump(word_counter, t)
t.close()
import operator
import pickle
t = open("mimic_wc.pickle", 'rb')
x = pickle.load(t)
t.close()
sorted_x = sorted(x.items(), key=operator.itemgetter(1))
print(sorted_x)


import pickle
import os
import inflect
inflect = inflect.engine()

f = open("allacronyms_consolidated_20190310.pickle", 'rb')
abbr_dict = pickle.load(f)
f.close()

src_dir = '/Users/Marta/80k_abbreviations/preprocess_pipeline'
#g = open(os.path.join(src_dir, "umls_name2id_20190310.pickle"), 'rb')
g = open(os.path.join(src_dir, "umls_name2id_20190402.pickle"), 'rb')
umls_name2id = pickle.load(g)
g.close()

umls_name2id["cancer"] = "C0006826"
umls_name2id["arterio venous"].add("C0450109")

def get_plural(term):
    return inflect.plural_noun(term)

cleaned_dict = {}
meta2name = {}
name2meta = {}
meta2cui = {}
raw_cui2meta = {}
cui2meta = {}

counter = 1
counter_ = 1
counter_2 = 1
for abbr in abbr_dict:
    if len(abbr_dict[abbr]) == 1:
        counter_ += 1
        continue
    plural_terms = []

    cleaned_dict[abbr] = {}
    meta2cui[abbr] = {}
    raw_cui2meta[abbr] = {}
    cui2meta[abbr] = {}
    name2meta[abbr] = {}
    meta2name[abbr] = {}

    terms = {}
    non_umls_terms = []
    abbr_id = 0
    for term in abbr_dict[abbr]:
        if term not in umls_name2id:
            continue
        #for cui in umls_name2id[term]:
        all_cuis = list(umls_name2id[term])
        present = False
        for cui in all_cuis:
            if not present and cui in raw_cui2meta[abbr]:
                meta_id = raw_cui2meta[abbr][cui]
                present = True

        if present:
            for cui in all_cuis:
                if cui not in terms[meta_id]:
                    terms[meta_id][cui] = set()
                terms[meta_id][cui].add(term)
        else:
            meta_id = abbr_id
            abbr_id += 1
            terms[meta_id] = {}
            for cui in all_cuis:
                if cui not in terms[meta_id]:
                    terms[meta_id][cui] = set()
                terms[meta_id][cui].add(term)
                if cui not in raw_cui2meta[abbr]:
                    raw_cui2meta[abbr][cui] = meta_id


    reduced_terms = {}
    for meta_id in terms:
        reduced_terms[meta_id] = {}
        if len(terms[meta_id]) > 1:
            cui_1 = sorted(terms[meta_id])[0]
            reduced_terms[meta_id][cui_1] = terms[meta_id][cui_1]
        else:
            reduced_terms[meta_id] = terms[meta_id]
    if len(reduced_terms) < 2:
        counter_2 += 1
        continue
    counter += 1
    cleaned_dict[abbr] = reduced_terms
    for meta_id in reduced_terms:
        cui = list(reduced_terms[meta_id].keys())[0]
        term = reduced_terms[meta_id][cui]
        cui2meta[abbr][cui] = meta_id
        meta2cui[abbr][meta_id] = cui
        for cui in  reduced_terms[meta_id]:
            for term in reduced_terms[meta_id][cui]:
                try:
                    meta2name[abbr][meta_id].add(term)
                except:
                    meta2name[abbr][meta_id] = set()
                    meta2name[abbr][meta_id].add(term)
                name2meta[abbr][term] = meta_id



        '''
        cui_1 = list(umls_name2id[term])[0]
        if cui_1 not in cui2meta[abbr]:
            meta_id = abbr_id
            terms[meta_id] = {}
            cui2meta[abbr][cui_1] = meta_id
            meta2cui[abbr][meta_id] = cui_1
            abbr_id += 1


        meta_id = cui2meta[abbr][cui_1]
        name2meta[abbr][term] = meta_id

        try:
            meta2name[abbr][meta_id].add(term)
        except:
            meta2name[abbr][meta_id] = set()
            meta2name[abbr][meta_id].add(term)

        try:
            terms[meta_id][cui_1].add(term)
        except:
            terms[meta_id][cui_1] = set()
            terms[meta_id][cui_1].add(term)


       '''
    '''
        for cui_ in umls_name2id[term]:
            cui2meta[abbr][cui_] = meta_id

            try:
                meta2cui[abbr][meta_id].add(cui_)
            except:
                meta2cui[abbr][meta_id] = set()
                meta2cui[abbr][meta_id].add(cui_)

            try:
                terms[meta_id][cui_].add(term)
            except:
                terms[meta_id][cui_] = set()
                terms[meta_id][cui_].add(term)

            name2meta[abbr][term] = meta_id
            try:
                meta2name[abbr][meta_id].add(term)
            except:
                meta2name[abbr][meta_id] = set()
                meta2name[abbr][meta_id].add(term)
        
    
    if len(terms) > 1:
        cleaned_dict[abbr] = terms
        counter += 1
    else:
        counter_2 +=1
    '''
print("There are " + str(counter)+ " abbbreviations with more than one expansion.")
print("There are " + str(counter_) + " abbreviations with one expansion.")
print("There are " + str(counter_2) + " abbreviations with one expansion after cleaning.")

#m = open("cleaned_allacronyms_dict_20190318.pickle", 'wb')
m = open("cleaned_allacronyms_dict_20190402_NEW_2.pickle", 'wb')
pickle.dump(cleaned_dict, m)
m.close()

#m = open("allacronyms_name2meta_20190318.pickle", 'wb')
m = open("allacronyms_name2meta_20190402_NEW_2.pickle", 'wb')
pickle.dump(name2meta, m)
m.close()

#m = open("allacronyms_meta2name_20190318.pickle", 'wb')
m = open("allacronyms_meta2name_20190402_NEW_2.pickle", 'wb')
pickle.dump(meta2name, m)
m.close()

#m = open("allacronyms_cui2meta_20190318.pickle", 'wb')
m = open("allacronyms_cui2meta_20190402_NEW_2.pickle", 'wb')
pickle.dump(cui2meta, m)
m.close()

#m = open("allacronyms_meta2cui_20190318.pickle", 'wb')
m = open("allacronyms_meta2cui_20190402_NEW_2.pickle", 'wb')
pickle.dump(meta2cui, m)
m.close()

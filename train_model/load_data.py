
from localglobalembed import AbbrRep
import predict_unlabelled_data
from sklearn.utils import shuffle
import math
from get_card_exp import get_card_expansions
import pickle
import os

def casi_process(data_dict, ns=100):
    key = "casi_abbr"
    casi_test = []
    for subkey in data_dict[key]:
        casi_test.extend(data_dict[key][subkey])
    casi_test = shuffle(casi_test, random_state=42)[:ns]
    return casi_test

def unlabelled(data_dict, key):
    unlabelled_train = []
    unlabelled_val = []
    for subkey in data_dict[key]:
        print(subkey)
        split = math.floor(len(data_dict[key][subkey])*0.7)
        unlabelled_train.extend(data_dict[key][subkey][:split])
        unlabelled_val.extend(data_dict[key][subkey][split:])

    return unlabelled_train, unlabelled_val

def reverse_sub_test(data_dict):
    key = "mimic_abbr"
    key_2 = "mimic_rs"
    X = []
    labels = set()
    for subkey in data_dict[key]:
        for item in data_dict[key][subkey]:
            labels.add(item.label)
    labels = list(labels)
    for subkey in labels:
        if subkey in data_dict[key_2]:
            X.extend(data_dict[key_2][subkey])
    ns = min(len(X), 100)
    X = shuffle(X, random_state=42)[:ns]
    return X

def reverse_sub(data_dict, key, abbr, cap=0, limit=False):
    mimic_train = []
    mimic_val = []

    if limit:
        card_exp = get_card_expansions()

    if cap == 0:
        for subkey in data_dict[key]:
            if limit and subkey in card_exp:
                continue
            print(subkey)

            if len(data_dict[key][subkey]) > 4:
                split = math.floor(len(data_dict[key][subkey])*0.7)
                mimic_train.extend(data_dict[key][subkey][:split])
                mimic_val.extend(data_dict[key][subkey][split:])
            else:
               mimic_train.extend(data_dict[key][subkey])
    else:
        moving_cap = cap
        under_rep = []
        over_rep = []
        for subkey in data_dict[key]:
            if limit and subkey not in card_exp[abbr]:
                continue
            if subkey == 16 or subkey == 22:
                continue
            if len(data_dict[key][subkey]) <= 15:
                under_rep.extend(data_dict[key][subkey])
                moving_cap -= len(data_dict[key][subkey])
            else:
                if subkey == 47:
                    counter = 0
                    for i in data_dict[key][subkey]:
                        if i.source in ['C0011633','C0263667','C3278909']:
                            counter += 1
                            continue
                        else:
                            over_rep.append(i)
                    print("COUNTER " + str(counter))

                over_rep.extend(data_dict[key][subkey])
        over_rep = shuffle(over_rep, random_state=42)[:moving_cap]
        full_data = over_rep
        full_data.extend(under_rep)
        full_data = shuffle(full_data, random_state=42)
        if len(full_data) >= 500:
            split = math.floor(cap*0.7)
        else:
            split = math.floor(len(full_data) * 0.7)
        mimic_train = full_data[:split]
        mimic_val = full_data[split:]

    return mimic_train, mimic_val


def reverse_sub_unlabelled(data_dict, abbr, cap=500, limit=False):
    key = "mimic_rs"
    key2 = "mimic_abbr"

    moving_cap = cap
    under_rep = []
    over_rep = []
    if limit:
        card_exp = get_card_expansions()

    for subkey in data_dict[key]:
        if limit and subkey not in card_exp[abbr]:
            continue
        if len(data_dict[key][subkey]) <= 15:
            under_rep.extend(data_dict[key][subkey])
            moving_cap -= len(data_dict[key][subkey])
        else:
            over_rep.extend(data_dict[key][subkey])
    for subkey in data_dict[key2]:
        if len(data_dict[key2][subkey]) <= 10:
            under_rep.extend(data_dict[key2][subkey])
            moving_cap -= len(data_dict[key2][subkey])
        else:
            over_rep.extend(data_dict[key2][subkey])

    over_rep = shuffle(over_rep, random_state=42)[:moving_cap]
    full_data = over_rep
    full_data.extend(under_rep)

    if len(full_data) >= 500:
        split = math.floor(cap * 0.7)
    else:
        split = math.floor(len(full_data) * 0.7)

    mimic_train = full_data[:split]
    mimic_val = full_data[split:]

    return mimic_train, mimic_val


def reverse_sub_unlabelled_augment(data_dict, abbr, cap=500, limit=False):
    key = "mimic_rs"
    key2 = "mimic_abbr"

    moving_cap = cap
    under_rep = []
    over_rep = []
    if limit:
        card_exp = get_card_expansions()
    for subkey in data_dict[key]:
        if limit and subkey not in card_exp[abbr]:
            continue
        if len(data_dict[key][subkey]) <= 15:
            under_rep.extend(data_dict[key][subkey])
            moving_cap -= len(data_dict[key][subkey])
        else:
            over_rep.extend(data_dict[key][subkey])


    unlabelled_data = {}
    for subkey in data_dict[key2]:
        for item in data_dict[key2][subkey]:
            if item.label not in data_dict[key] or len(data_dict[key][item.label]) < 3:
                try:
                    unlabelled_data[item.label].append(item)
                except KeyError:
                    unlabelled_data[item.label] = [item]

    for key in unlabelled_data:
        data = shuffle(unlabelled_data[key], random_state=42)
        if abbr == "cea":
            under_rep.extend(data[:50])
        else:

            under_rep.extend(data[:3])

    over_rep = shuffle(over_rep, random_state=42)[:moving_cap]
    full_data = over_rep
    full_data.extend(under_rep)
    full_data = shuffle(full_data, random_state=42)
    if len(full_data) >= 500:
        split = math.floor(cap * 0.7)
    else:
        split = math.floor(len(full_data) * 0.7)
    mimic_train = full_data[:split]
    mimic_val = full_data[split:]

    return mimic_train, mimic_val

def reverse_sub_sim(data_dict):
    mimic_train = []
    mimic_val = []
    key = "mimic_rs"
    key2 = "mimic_rs_sim"
    for subkey in data_dict[key]:
        combo = data_dict[key][subkey]
        if subkey in data_dict[key2]:
            combo += data_dict[key2][subkey]
        if len(combo) > 4:
            combo = shuffle(combo, random_state=42)
            split = math.floor(min(500,len(combo))*0.7)
            mimic_train.extend(combo[:split])
            mimic_val.extend(combo[split:])
        else:
            mimic_train.extend(combo)

    return mimic_train, mimic_val
'''
def reverse_sub_unlabelled(data_dict, method):
    mimic_train = []
    mimic_val = []
    key = "mimic_rs"
    key2 = "mimic_abbr"
    unlabelled_data = {}

    unlabelled_x = []
    original_x = []
    for subkey in data_dict[key2]:
        for i in data_dict[key2][subkey]:
            unlabelled_x.append(i.embedding[0])
            original_x.append(i)

    if method == "gda":
        unlabelled_predictions = predict_unlabelled_data.predict_labels(data_dict, unlabelled_x)

    elif method == "knn":
        unlabelled_predictions = predict_unlabelled_data.knn(data_dict, unlabelled_x)

    for i in range(len(unlabelled_predictions)):
        y = unlabelled_predictions[i]
        x = original_x[i]
        try:
            unlabelled_data[y].append(x)
        except KeyError:
            unlabelled_data[y] = [x]

    for subkey in data_dict[key]:
        combo = data_dict[key][subkey]
        if subkey in unlabelled_data:
            combo += unlabelled_data[subkey]
        if len(combo) > 4:
            combo = shuffle(combo, random_state=42)
            split = math.floor(min(500,len(combo))*0.7)
            mimic_train.extend(combo[:split])
            mimic_val.extend(combo[split:])
        else:
            mimic_train.extend(combo)

    return mimic_train, mimic_val

def reverse_sub_sim_unlabelled(data_dict, method):
    mimic_train = []
    mimic_val = []
    key = "mimic_rs"
    key2 = "mimic_abbr"
    unlabelled_data = {}
    key3 = "mimic_rs_sim"

    unlabelled_x = []
    original_x = []
    for subkey in data_dict[key2]:
        for i in data_dict[key2][subkey]:
            unlabelled_x.append(i.embedding[0])
            original_x.append(i)

    if method == "gda":
        unlabelled_predictions = predict_unlabelled_data.predict_labels(data_dict, unlabelled_x, sim=True)

    elif method == "knn":
        unlabelled_predictions = predict_unlabelled_data.knn(data_dict, unlabelled_x, sim=True)

    for i in range(len(unlabelled_predictions)):
        y = unlabelled_predictions[i]
        x = original_x[i]
        try:
            unlabelled_data[y].append(x)
        except KeyError:
            unlabelled_data[y] = [x]

    for subkey in data_dict[key]:
        combo = data_dict[key][subkey]
        if subkey in unlabelled_data:
            combo += unlabelled_data[subkey]
        if subkey in data_dict[key3]:
            combo += data_dict[key3][subkey]
        if len(combo) > 4:
            combo = shuffle(combo, random_state=42)
            split = math.floor(min(500,len(combo))*0.7)
            mimic_train.extend(combo[:split])
            mimic_val.extend(combo[split:])
        else:
            mimic_train.extend(combo)

    return mimic_train, mimic_val
'''

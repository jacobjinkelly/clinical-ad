from abbrrep_class import AbbrRep
# import predict_unlabelled_data
from sklearn.utils import shuffle
import math

def casi_process(data_dict, ns=100):
    key = "casi_abbr"
    casi_test = []
    for subkey in data_dict[key]:
        casi_test.extend(data_dict[key][subkey])
    casi_test = shuffle(casi_test, random_state=42)
    return casi_test

def unlabelled(data_dict, key):
    unlabelled_train = []
    unlabelled_val = []

    for subkey in data_dict[key]:
        split = math.floor(len(data_dict[key][subkey])*0.7)
        unlabelled_train.extend(data_dict[key][subkey][:split])
        unlabelled_val.extend(data_dict[key][subkey][split:])

    return unlabelled_train, unlabelled_val

def reverse_sub(data, opt):
    full_data = []
    for subkey in data["mimic_rs"]:
        curr_data = data["mimic_rs"][subkey]
        curr_data = shuffle(curr_data, random_state=42)
        full_data.extend(curr_data[:min(opt.ns, len(data["mimic_rs"][subkey]))])
    full_data = shuffle(full_data, random_state=42)
    split = math.floor(len(full_data) * 0.7)
    mimic_train = full_data[:split]
    mimic_val = full_data[split:]

    return mimic_train, mimic_val


def reverse_sub_unlabelled(data_dict, cap=500):
    key = "mimic_rs"
    key2 = "mimic_abbr"

    moving_cap = cap
    under_rep = []
    over_rep = []
    for subkey in data_dict[key]:
        if len(data_dict[key][subkey]) <= 10:
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
    split = math.floor(cap * 0.7)
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

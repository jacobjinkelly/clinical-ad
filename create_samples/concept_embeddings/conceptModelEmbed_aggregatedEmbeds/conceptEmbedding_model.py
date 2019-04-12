from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from abbrrep_class import AbbrRep
import pickle
import traceback
import sys
import scipy.sparse
import argparse
import random
import datetime
import os
random.seed(16)
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")


def load_data(opt):
    #src = "/Volumes/terminator/hpf/"
    #src = "/hpf/projects/brudno/marta/mimic_rs_collection/concept_embed_model"
    
    data = []
    fname = opt.inputfile.split(",")
    for i in fname:
        with open(i, 'rb') as file_handle:
            print("file loaded: " + i)
            dict = pickle.load(file_handle)
            for key in dict:
                data.extend(dict[key])
            print(len(data))
    random.shuffle(data)
    return data[:4000]
    '''
    file_hash = {0: "conceptembeds50_w5_20190317_10.pickle", 1: "conceptembeds50_w5_20190317_1.pickle",
                 2: "conceptembeds50_w5_20190317_2.pickle", 3: "conceptembeds50_w5_20190317_3.pickle",
                 4: "conceptembeds50_w5_20190317_4.pickle", 5: "conceptembeds50_w5_20190317_5.pickle",
                 6: "conceptembeds50_w5_20190317_6.pickle", 7: "conceptembeds50_w5_20190317_7.pickle",
                 8: "conceptembeds50_w5_20190317_8.pickle", 9: "conceptembeds50_w5_20190317_9.pickle"}
    
    #pickle_in = open(os.path.join(src, file_hash[file_num]), 'rb')
    data = []
    for i in range(10):
        f = open(file_hash[], 'rb')
        data.extend(pickle.load(f))
        f.close()
            
    print(len(data))
    return data

   
    with open(file_hash[file_num], 'rb') as file_handle:
        print("file loaded: " + file_hash[file_num])
        data = pickle.load(file_handle)
        return data
    '''
class Data(Dataset):
    def __init__(self, X, word2idx):
        self.X=np.array(X)
        self.word2idx = word2idx
    def __len__(self):
        return len(self.X)
    def __getitem__(self, id):
        X = torch.tensor(self.X[id].embedding)
        print(self.X[id].label)
        y = self.word2idx[self.X[id].source]
        return X, y


def get_sparseMatrix(opt):
    
    p_in = open(opt.sm, 'rb')
    umls_rel = pickle.load(p_in)
    p_in.close()

    matrix_hierarchy = umls_rel["matrix_hierarchy"]
    word2idx = umls_rel["word2idx"]
    idx2word = umls_rel["idx2word"]

    return matrix_hierarchy, word2idx, idx2word



class ConceptEmbedModel(nn.Module):
    def __init__(self, a, embedding_dim, pretrained_weight):
        super().__init__()
        self.embed1 = nn.Embedding(a, embedding_dim)
        self.embed1.weight.data.copy_(torch.from_numpy(pretrained_weight))
	        
    def forward(self, ancestor_idx):
        print(self.embed1(ancestor_idx))
        running_ancestor_embeds_sum = self.embed1(ancestor_idx).sum(1)
        return running_ancestor_embeds_sum


def save_output(net, idx2word, word2idx, trainLoss_array, valLoss_array, opt, start, error=None, checkpoint=None):
    for child in net.children():
        for param in child.parameters():
            trained_weights = param.detach().cpu().numpy()

            concept_embedding_weights = {"weights": trained_weights,
                                         "idx2word": idx2word,
                                         "word2idx": word2idx,
                                         "trainLoss": trainLoss_array,
                                         "valLoss": valLoss_array,
                                         "params": opt,
                                         "start": start,
                                         "end": datetime.datetime.now(),
                                         "notes": "10 epochs",
                                         "error": error}
            if checkpoint:
                f = opt.abbr + "_checkpoint" + str(checkpoint) + ".pickle"
            else:
                f = opt.outputfile
            p_out = open(f, "wb")
            pickle.dump(concept_embedding_weights, p_out)
            p_out.close()


def train_model(opt, pretrained_weight, word2idx, idx2word, matrix_hierarchy, concepts):
    start = datetime.datetime.now()
    net = ConceptEmbedModel(len(concepts) + 1, 200, pretrained_weight)
    net = net.to(device)
    try:
        criterion = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=float(opt.lr), momentum=float(opt.momentum))
        #optimizer = optim.SGD(net.parameters(), lr=float(opt.lr), momentum=float(opt.momentum), nesterov=True)
        #optimizer = optim.Adam(net.parameters(), lr=float(opt.lr))
        max_epochs = int(opt.max_epochs)
        trainLoss_array = []
        valLoss_array = []

        data = load_data(opt)
        params = {'batch_size': int(opt.batch_size),
                  'shuffle': True,
                  'num_workers': int(opt.num_workers)}

        train_X, val_X = train_test_split(data, test_size=float(opt.test_size), random_state=42)
        del(data)
        training_set = Data(train_X, word2idx)
        training_generator = DataLoader(training_set, **params)

        validation_set = Data(val_X, word2idx)
        validation_generator = DataLoader(validation_set, **params)

        for epoch in range(max_epochs):  # loop over the dataset multiple times
            running_trainLoss = 0.0
            running_valLoss = 0.0
            
            '''
            data = load_data(epoch % 10)

            params = {'batch_size': int(opt.batch_size),
                      'shuffle': True,
                      'num_workers': int(opt.num_workers)}

            train_X, val_X = train_test_split(data, test_size=float(opt.test_size), random_state=42)
            del(data)
            training_set = Data(train_X, word2idx)
            training_generator = DataLoader(training_set, **params)

            validation_set = Data(val_X, word2idx)
            validation_generator = DataLoader(validation_set, **params)
            
            del(train_X)
            del(val_X)
            del(training_set)
            del(validation_set)
            '''
            
            # for sample in data:
            for local_batch, local_labels in tqdm(training_generator):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                # zero the parameter gradients
                
                
                # get ancestors
                cuis = local_labels.cpu().tolist()
                ancestors = matrix_hierarchy.rows[cuis]

                inputs = [torch.LongTensor(ancestors[i]) for i in range(len(ancestors))]
                #inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
             
                output = net(torch.stack(inputs).to('cuda'))
                #output = net(inputs.type(torch.LongTensor).to(device))
                
                X = local_batch.clone().detach()
                X = X.view(-1,200)
                loss = criterion(output, X.type(torch.FloatTensor).to(device))
                #loss = criterion(output, X.type(torch.LongTensor).to(device))

                
                loss.backward()
                
                for child in net.children():
                    for param in child.parameters():
                        param.grad[0, :] = 0
                
                optimizer.step()
                optimizer.zero_grad()
                net.zero_grad()
                

                # print statistics
                running_trainLoss += (loss.item())

            

            with torch.set_grad_enabled(False):
                for local_batch, local_labels in validation_generator:
                    net.zero_grad()
                    optimizer.zero_grad()
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                    
                    cuis = local_labels.cpu().tolist()
                    ancestors = matrix_hierarchy.rows[cuis]
                    inputs = [torch.LongTensor(ancestors[i]) for i in range(len(ancestors))]
                    output = net(torch.stack(inputs).to('cuda'))
                    #inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
                    #output = net(inputs.type(torch.LongTensor).to(device))
                    X = local_batch.clone().detach()
                    
                    X = X.view(-1,200)
                    loss = criterion(output, X.type(torch.FloatTensor).to(device))
                    #loss = criterion(output, X.type(torch.LongTensor).to(device))
                    running_valLoss += loss.item()
                    # if i % 2000 == 1999:    # print every 2000 mini-batches
     

            trainLoss_array.append(running_trainLoss / len(train_X))
            valLoss_array.append(running_valLoss / len(val_X))
            
            if epoch % 10 == 0:
                date = datetime.datetime.today().strftime('%Y%m%d')
                curr_time = datetime.datetime.now()
                time = str(curr_time.hour) + "-" + str(curr_time.minute)
                fname = date + "_" + time + "_epoch" + str(epoch) + "_checkpoint.pickle"
                save_output(net, idx2word, word2idx, trainLoss_array, valLoss_array, opt, start, error=None, checkpoint=epoch)
                '''
                for child in net.children():
                    for param in child.parameters():
                        trained_weights = param.detach().cpu().numpy()
                        f = open(fname, 'wb')
                        pickle.dump(trained_weights, f)
                        f.close()
                '''
            print('Epoch [%d] training loss: %.6f' % (epoch + 1, running_trainLoss / len(train_X)))
            print('Epoch [%d] validation loss: %.6f' % (epoch + 1, running_valLoss / len(val_X)))
            print(datetime.datetime.now())
    
        print('Finished Training')
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        save_output(net, idx2word, word2idx, trainLoss_array, valLoss_array,opt, start, error=[traceback.format_exc(),e])
    finally:
        save_output(net, idx2word, word2idx, trainLoss_array, valLoss_array,opt, start)
    
def setup_model(opt):

    matrix_hierarchy, word2idx, idx2word = get_sparseMatrix(opt)
    concepts = word2idx
    #pretrained_weight = np.random.uniform(low=-1.0e-9, high=1.0e-9, size=(len(concepts) + 1, 200))
    pretrained_weight = np.zeros((len(concepts)+1, 200))
    train_model(opt, pretrained_weight, word2idx, idx2word, matrix_hierarchy, concepts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputfile', required=True)
    parser.add_argument('-sm', required=True)
    parser.add_argument('-abbr', required=True)
    parser.add_argument('-max_epochs', default=1000, help="starting index of cui files to generate embeddings "
                                                   "from; e.g.: '1'")
    parser.add_argument('-batch_size', default=256, help="max number of words to consider in local_context")
    parser.add_argument('-num_workers', default=8, help="max number of random samples PER EXPANSION to "
                                                 "create training set from")
    parser.add_argument('-test_size', default=0.3, help="max number of random samples PER EXPANSION to "
                                                        "create training set from")
    parser.add_argument('-lr', default=0.01, help="max number of random samples PER EXPANSION to "
                                                        "create training set from")

    parser.add_argument('-momentum', default=0.9, help="max number of random samples PER EXPANSION to "
                                                   "create training set from")

                                                   
    opt = parser.parse_args()
    date = datetime.datetime.today().strftime('%Y%m%d')
    curr_time = datetime.datetime.now()
    time = str(curr_time.hour) + "-" + str(curr_time.minute) 
    fname = opt.abbr + "_4000s_" + date  + "_CONCEPTMODEL.pickle"
    print("Output file name: " + fname)
    parser.add_argument('-outputfile', default="", help="max number of random samples PER EXPANSION to "
                                                       "create training set from")

    opt.outputfile = fname
    setup_model(opt)

    print("Done training concept embed model!!! \U0001F388 \U0001F33B")
    torch.cuda.empty_cache()
if __name__ == "__main__":
    main()

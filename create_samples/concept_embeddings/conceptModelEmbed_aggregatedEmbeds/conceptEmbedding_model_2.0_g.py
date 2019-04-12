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
    return data

class Data(Dataset):
    def __init__(self, X, word2idx):
        self.X=[]
        self.word2idx = word2idx
        for i in range(len(X)):
            if X[i].source not in self.word2idx:
                continue
            self.X.append(X[i])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, id):
        X = torch.tensor(self.X[id].embedding)
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
    def __init__(self, a, embedding_dim, pretrained_weight, ancestor_hierarchy, concepts):
        super().__init__()
        self.embed1 = nn.Embedding(a, embedding_dim)
        self.concepts = concepts
        self.embed1.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.ancestor_hierarchy = ancestor_hierarchy
        self.H = Variable(torch.randn(len(self.concepts), 200, dtype=torch.double) * 1e-9, requires_grad=False)
    
    def forward(self, x, y):
        
        local_H = torch.zeros(len(self.concepts), 200, dtype=torch.double, requires_grad=False)
        for i in range(len(self.concepts)):
            running_ancestor_embeds = self.embed1(torch.tensor(self.ancestor_hierarchy.rows[i]).to(device))
            running_ancestor_embeds_sum = running_ancestor_embeds.sum(0)
            local_H[i, :] = running_ancestor_embeds_sum
        

        concept_vector = torch.mm(local_H.to(device), torch.t(x.squeeze(1)).to(device))  # C * 200 x 200 * len(x)
        m = nn.Softmax(dim=1)
        x = torch.t(concept_vector)
        output = m(x)
        return output




def train_model(opt, pretrained_weight, word2idx, idx2word, matrix_hierarchy, concepts):
    start = datetime.datetime.now()
    net = ConceptEmbedModel(len(concepts), 200, pretrained_weight, matrix_hierarchy, concepts)
    net = net.to(device)
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=float(opt.lr), momentum=float(opt.momentum))
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

            
            # for sample in data:
            for local_batch, local_labels in tqdm(training_generator):

                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                output = net(local_batch, local_labels)
                values, indices = output.max(1)
                loss = criterion(output, torch.tensor(local_labels, dtype=torch.long).to(device))
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                net.zero_grad()
                
                # print statistics
                running_trainLoss += (loss.item())

            

            with torch.set_grad_enabled(False):
                for local_batch, local_labels in validation_generator:
                    net.zero_grad()
                    optimizer.zero_grad()

                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                    output = net(local_batch, local_labels)
                    loss = criterion(output, torch.tensor(local_labels, dtype=torch.long).to(device))
                    running_valLoss += loss.item()
                    # if i % 2000 == 1999:    # print every 2000 mini-batches
     

            trainLoss_array.append(running_trainLoss / len(train_X))
            valLoss_array.append(running_valLoss / len(val_X))
            
            if (epoch+1) % 5 == 0:
                date = datetime.datetime.today().strftime('%Y%m%d')
                filename = opt.abbr + "_" + date + "_epoch" + str(epoch+1)
                torch_filename = filename+ ".pth.tar"
                text_filename = filename + ".txt"
                curr_dir = "/home/marta/conceptEmbedModel/20190403_models_w5_g"
                torch.save(net.state_dict(), os.path.join(curr_dir,torch_filename))
                
                f = open(os.path.join(curr_dir,text_filename), 'a')
                curr_time = datetime.datetime.now()
                f.write(str(curr_time))
                f.write("Train")
                f.write(str(trainLoss_array))
                f.write("Val")
                f.write(str(valLoss_array))
                f.write(str(opt))
                f.close()
                print('Saved: '+ str(epoch+1))
            
            print('Epoch [%d] training loss: %.6f' % (epoch + 1, running_trainLoss / len(train_X)))
            print('Epoch [%d] validation loss: %.6f' % (epoch + 1, running_valLoss / len(val_X)))
            print(datetime.datetime.now())
    
        print('Finished Training')
    except Exception as e:
        print(traceback.format_exc())
        print(e)

    
def setup_model(opt):

    matrix_hierarchy, word2idx, idx2word = get_sparseMatrix(opt)
    concepts = word2idx
    pretrained_weight = np.random.uniform(low=-1.0e-9, high=1.0e-9, size=(len(concepts), 200))
    #pretrained_weight = np.zeros((len(concepts), 200))
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

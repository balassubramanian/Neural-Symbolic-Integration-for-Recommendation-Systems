import torch
import torch.nn.functional as F
# from torch import linalg as LA

import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from scipy.sparse import csr_matrix, rand as sprand
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Edit the path as per your embedding location
logical_embedding_path = '/Users/balassubramaniansrinivasan/Downloads/project6/MF_RMSE/Logical embeddings/logical_embeddings_hul.pt'
alpha_seq_path = '/Users/balassubramaniansrinivasan/Downloads/project6/MF_RMSE/Logical embeddings/alpha_seq.pt'
beta_seq_path = '/Users/balassubramaniansrinivasan/Downloads/project6/MF_RMSE/Logical embeddings/beta_seq.pt'
alpha_output_path = '/Users/balassubramaniansrinivasan/Downloads/project6/MF_RMSE/Logical embeddings/alpha_output.pt'
beta_output_path = '/Users/balassubramaniansrinivasan/Downloads/project6/MF_RMSE/Logical embeddings/beta_output.pt'

#logical_embedding_path = torch.load(logical_embedding_pathh)
#logical_embedding_path = logical_embedding_path.to(device)

class MF(torch.nn.Module):
    def __init__(self, arguments, train_df, train_like, test_like,logical_embedding_path,alpha_seq_path,beta_seq_path,alpha_output_path,beta_output_path):
        super(MF, self).__init__()

        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.learning_rate = arguments['learning_rate']
        self.epochs = arguments['epoch']
        self.display = arguments['display']
        self.regularization = arguments['reg']
        self.hidden = arguments['hidden']
        self.neg_sampling = arguments['neg']
        self.data = arguments['data']
        self.batch_size = arguments['bs']

        self.train_df = train_df
        self.train_like = train_like
        self.test_like = test_like

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")


        #############################################################
        
        self.logical_embedding_path = logical_embedding_path
        self.alpha_seq_path = alpha_seq_path
        self.beta_seq_path = beta_seq_path
        self.alpha_output_path = alpha_output_path
        self.beta_output_path = beta_output_path
        self.logical_embeddings = torch.load(self.logical_embedding_path)
        self.alpha_seq = torch.load(alpha_seq_path)
        self.beta_seq = torch.load(beta_seq_path)
        self.alpha_output = torch.load(alpha_output_path)
        self.beta_output = torch.load(beta_output_path)

        ############################################################

        print('******************** MF ********************')

        self.user_factors = torch.nn.Embedding(self.num_users, self.hidden)
        self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors = torch.nn.Embedding(self.num_items, self.hidden)
        self.item_factors.weight.data.uniform_(-0.05, 0.05)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        self.loss_function = torch.nn.MSELoss()

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user, item):
        # Get the dot product per row

        logical_embeddings_2d = self.logical_embeddings.mean(dim=1)
        alpha_seq = self.alpha_seq.mean(dim=1)
        #print('alpha_seq',alpha_seq.shape)
        beta_seq = self.beta_seq.mean(dim=1)
        u_logic = alpha_seq / (alpha_seq+beta_seq)

        u = self.user_factors(user)
        v = self.item_factors(item)
        #print('u',u.shape)
        #print('v',v.shape)
        u_logic = torch.repeat_interleave(u_logic, 2, dim=1)
        #print('u_logic',u_logic.shape)
        #u_combined = torch.cat([u, u_logic], dim=1)
        #print('ucombined',u_combined.shape)
        #x = (u * v * u_logic ).sum(axis=1)
        x = (u * v ).sum(axis=1)
        #print('x',x.shape)
        return x

    def train_model(self, itr):
        self.train()
        epoch_cost = 0.
        self.user_list, self.item_list, self.label_list = utility.negative_sampling(self.num_users, self.num_items,
                                                                                    self.train_df['userId'].values,
                                                                                    self.train_df['itemId'].values,
                                                                                    self.neg_sampling)
        start_time = time.time() * 1000.0
        #num_batch = int(len(self.user_list) / float(self.batch_size)) + 1
        num_batch = int(len(self.user_list) / 1024)
        random_idx = np.random.permutation(len(self.user_list))

        for i in tqdm(range(num_batch)):
            '''''
            batch_idx = None
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            '''''
            batch_idx = random_idx[i * 1024 : (i + 1) * 1024] 
            tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                                                              self.label_list[batch_idx])

            epoch_cost += tmp_cost

        print("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
              "Training time : %d ms" % (time.time() * 1000.0 - start_time))

    def train_batch(self, user_input, item_input, label_input):
        # reset gradients
        self.optimizer.zero_grad()

        users = torch.Tensor(user_input).long().to(self.device)
        items = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)

        y_hat = self.forward(users, items)
        loss = self.loss_function(y_hat, labels)

        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item()


    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        self.eval()

        start_time = time.time() * 1000.0
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().numpy()
        Q = Q.detach().numpy()
        Rec = np.matmul(P, Q.T)

        precision, recall, ndcg = utility.MP_test_model_all(Rec, self.test_like, self.train_like, n_workers=10)

        print("Testing //", "Epoch %d //" % itr,
              "Accuracy Testing time : %d ms" % (time.time() * 1000.0 - start_time))
        print("=" * 100)
        return np.mean(ndcg)


    def run(self):
        best_metric = -1
        best_itr = 0
        for epoch_itr in range(1, self.epochs + 1):
            self.train_model(epoch_itr)
            if epoch_itr % self.display == 0:
                cur_metric = self.test_model(epoch_itr)
                if cur_metric > best_metric:
                    best_metric = cur_metric
                    best_itr = epoch_itr
                    self.make_records(epoch_itr)
                elif epoch_itr - best_itr >= 30:
                    break

    def make_records(self, itr):  # record all the results' details into files
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        # with open('./' + self.args.data + '/P_MF_' + str(self.reg) + '.npy', "wb") as f:
        #     np.save(f, P)
        # with open('./' + self.args.data + '/Q_MF_' + str(self.reg) + '.npy', "wb") as f:
        #     np.save(f, Q)
        return P, Q


if __name__ == '__main__':
    
    args = {
        'epoch': 100,
        'display': 1,
        'learning_rate': 0.001,
        'reg': 0.000001,
        'hidden': 128,
        'neg': 2,
        'bs': 1024,
        'data': 'ML1M'
    }

    with open('./ML1M/info.pkl', 'rb') as f:
        info = pickle.load(f)
        args['num_user'] = info['num_user']
        args['num_item'] = info['num_item']
    print(info)

    train_like = list(np.load('./ML1M/user_train_like.npy', allow_pickle=True))
    test_like = list(np.load('./ML1M/user_vali_like.npy', allow_pickle=True))
    train_df = pd.read_csv('./ML1M/train_df.csv')

    model = MF(args, train_df, train_like, test_like,logical_embedding_path,alpha_seq_path, beta_seq_path, alpha_output_path,beta_output_path)
    model.run()

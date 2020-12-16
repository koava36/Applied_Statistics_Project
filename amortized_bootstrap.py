import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import resample


def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    return indices

def Amortized_bootstrap_learning(implicit_model, x_model, criterion, X_0, y_0, n_bootstrap_sampling,
                                 n_epochs, batch_size, learning_rate=0.1):
    # This function takes implicit model, data model p(y|x, theta) (x_model)
    # and data X_0, y_0
    # It learns the parameters of implicit model to be close to bootstrap distribution of theta
    
    N_bootstrap_sampling = n_bootstrap_sampling
    N_epochs = n_epochs  # Number of epochs with one bootstrap sample
    Batch_size = batch_size
    learning_rate = learning_rate
    n_train, n_features = X_0.shape # important! n_features includes bias 

    X_train, y_train = X_0, y_0
    f = implicit_model
    optimizer = optim.SGD(f.parameters(), lr=learning_rate)
    criterion = criterion

    for i in range(N_bootstrap_sampling):
        # sample once every few epochs
        indices = get_bootstrap_samples(X_train, 1)
        X_bs = X_train[indices[0]] # [n_train, n_features]
        y_bs = y_train[indices[0]] # [n_train, ]

        for epoch in range(N_epochs):
            mean_loss = np.array([])
            n = len(X_bs)
            # permutation is needed for different data on different epochs
            idx_perm = np.random.permutation(np.arange(n))
            X_bs = X_bs[idx_perm] # [n_train, _features]
            y_bs = y_bs[idx_perm] # [n_train, 1]

            for j in range(0, n, Batch_size):
                X_batch = X_bs[j:j + Batch_size] # [batch_size, _features]
                y_batch = y_bs[j: j + Batch_size] # [batch_size, ]

                # sample for each minibatch
                optimizer.zero_grad()
                
                ksi = np.random.normal(size=(1, 1))
                theta = f(torch.tensor(ksi).double())
                y_pred = x_model(torch.tensor(X_batch), theta) 
                loss = criterion(y_pred, torch.tensor(y_batch.reshape(-1, 1)).double())
                
                loss.backward()
                optimizer.step()
                
                mean_loss = np.append(mean_loss, loss.data.item())
        if i % 10 == 0:        
            print('Loss: ', mean_loss.mean())
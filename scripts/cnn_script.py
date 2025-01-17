import numpy as np
import pandas as pd
import scipy
import pickle
import torch
from torch import nn, optim
from torch.utils import data
from pyriemann.utils.mean import mean_covariance
import torch.nn.functional as F
import sklearn.datasets
import sklearn.decomposition

np.seterr(divide='ignore', invalid='ignore')


class Net(nn.Module):
    def __init__(self, nHidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(nHidden, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (3, 3)))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), (3, 3)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_data(parc, ref='none'):
    '''
    Navigates through file tree and extracts FCs with optional reconstruction
    '''
    if ref == 'none':
        with open(f'../data/schaefer/schaefer{parc}.pickle', 'rb') as f:
            all_FC = pickle.load(f)
    else:
        with open(f'../data/tangent_fcs/schaefer/schaefer{parc}_{ref}.pickle', 'rb') as f:
            all_FC = pickle.load(f)
    nSubj = int(all_FC.shape[0] / 16)
    nFCs = int(all_FC.shape[0])
    return all_FC, nSubj, nFCs


def q1invm(q1, eig_thresh=0):
    q1 += np.eye(q1.shape[0])
    U, S, V = scipy.linalg.svd(q1)
    S = np.diag(S ** (-1 / 2))
    Q1_inv_sqrt = U @ S @ V
    Q1_inv_sqrt = (Q1_inv_sqrt + np.transpose(Q1_inv_sqrt)) / 2
    return Q1_inv_sqrt

def qlog(q):
    U, S, V = scipy.linalg.svd(q)
    s = np.diag(S)
    S = np.diag(np.log(s))
    Q = U * S * np.transpose(V)
    return Q


def tangential(all_FC, ref):
    # Regularization for riemann
    if ref in ['riemann', 'kullback_sym', 'logeuclid']:
        print("Adding regularization!")
        eye_mat = np.eye(all_FC.shape[1])
        scaling_mat = np.repeat(eye_mat[None, ...], all_FC.shape[0], axis=0)
        all_FC += scaling_mat
    Cg = mean_covariance(all_FC, metric=ref)
    Q1_inv_sqrt = q1invm(Cg)
    Q = Q1_inv_sqrt @ all_FC @ Q1_inv_sqrt
    tangent_FC = np.empty(Q.shape)
    for idx, fc in enumerate(Q):
        if idx % 100 == 0:
            print(f'{idx}/{Q.shape[0]}')

        tangent_FC[idx] = linalg.logm(fc)
        enablePrint()
    return tangent_FC


def prepare_data(all_FC, nSubj, nFCs):
    '''
    Prepares labels and train, val and test data from raw data
    '''
    # Create labels corresponding to all_FC
    labels = torch.tensor(
        np.tile(np.repeat(np.arange(0, 8), nSubj), 2), dtype=torch.long)
    # Randomly shuffled indices for test FCs
    indices = np.random.permutation(nFCs)
    # Take subsets of data for training, validation, test
    train_val_idx = indices[:int(0.8 * nFCs)]

    val_idx = train_val_idx[int(0.8 * train_val_idx.shape[0]):]
    train_idx = train_val_idx[:int(0.8 * train_val_idx.shape[0])]
    test_idx = indices[int(0.8 * nFCs):]

    # Normalize data to training mean and stdev
    train_mean = np.mean(all_FC[train_idx])
    train_std = np.std(all_FC[train_idx])
    train_data = torch.FloatTensor(
        (all_FC[train_idx] - train_mean) / train_std)
    val_data = torch.FloatTensor(
        (all_FC[val_idx] - train_mean) / train_std)
    test_data = torch.FloatTensor(
        (all_FC[test_idx] - train_mean) / train_std)

    # Add channel dimension
    train_data = train_data.view(
        train_data.shape[0], -1, train_data.shape[1], train_data.shape[2])
    val_data = val_data.view(
        val_data.shape[0], -1, val_data.shape[1], val_data.shape[2])
    test_data = test_data.view(
        test_data.shape[0], -1, test_data.shape[1], test_data.shape[2])

    train_dataset = data.TensorDataset(
        train_data, labels[train_idx])  # create your datset
    val_dataset = data.TensorDataset(
        val_data, labels[val_idx])  # create your datset
    test_dataset = data.TensorDataset(
        test_data, labels[test_idx])  # create your datset

    train_loader = data.DataLoader(
        train_dataset, batch_size=80)  # create your dataloader
    val_loader = data.DataLoader(
        val_dataset, batch_size=80)  # create your dataloader
    test_loader = data.DataLoader(
        test_dataset, batch_size=80)  # create your dataloader
    return train_loader, val_loader, test_loader


def build_model(lr, nHidden):
    '''
    Given layer sizes and learning rate, builds model.
    Can change NN architecture here directly in nn.Sequential
    '''
    model = Net(nHidden)
    if use_cuda:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    return model, loss_fn, opt, history


def train_model(model, opt, loss_fn, train_loader, val_loader,
                max_epochs, n_epochs_stop, history):
    '''
    Trains model with specified parameters and returns trained model
    '''
    early_stop = False
    min_val_loss = np.Inf
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            if use_cuda:
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)

            opt.zero_grad()
            output = model(local_batch)
            loss = loss_fn(output, local_labels)
            loss.backward()
            opt.step()

            train_loss += loss.data.item() * local_batch.size(0)
            num_train_correct += (torch.max(output, 1)
                                  [1] == local_labels).sum().item()
            num_train_examples += local_batch.shape[0]

        train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in val_loader:
                # Transfer to GPU
                if use_cuda:
                    local_batch, local_labels = local_batch.to(
                        device), local_labels.to(device)
                output = model(local_batch)
                loss = loss_fn(output, local_labels)

                val_loss += loss.data.item() * local_batch.size(0)
                num_val_correct += (torch.max(output, 1)
                                    [1] == local_labels).sum().item()
                num_val_examples += local_batch.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(val_loader.dataset)

            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
        # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                early_stop = print('Early stopping!')
                break
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1} / {max_epochs},'
                  f'train loss: {train_loss: 5.4f},'
                  f'train acc: {train_acc: 5.3f}, val loss: {val_loss: 5.3f},'
                  f'val acc: {val_acc: 5.3f}')

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if early_stop:
            print("Stopped")
            break
    return model, history


def pca_recon(FC, pctComp=None):
    '''
    Reconstructs FC based on number of principle components
    '''
    if pctComp is None:
        return FC
    nRegions = FC.shape[1]
    FC = np.reshape(FC, (FC.shape[0], -1))
    nComp = int(FC.shape[0] * pctComp)
    mu = np.mean(FC, axis=0)
    pca_rest = sklearn.decomposition.PCA()
    pca_rest.fit(FC)
    SCORES = pca_rest.transform(FC)[:, :nComp]
    COEFFS = pca_rest.components_[:nComp, :]
    FC_recon = np.dot(SCORES, COEFFS)
    del SCORES, COEFFS
    FC_recon += mu
    FC_recon = np.reshape(FC_recon, (FC.shape[0], nRegions, nRegions))
    return FC_recon


def test_model(model, test_loader):
    '''
    After trained model is returned, this function tests the accuracy
    on novel data. Returns test accuracy of a model.
    '''
    model.eval()
    num_correct = 0
    num_examples = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_loader:
            # Transfer to GPU
            if use_cuda:
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)
            output = model(local_batch)
            num_correct += (torch.max(output, 1)
                            [1] == local_labels).sum().item()
            num_examples += local_batch.shape[0]

        test_acc = num_correct / num_examples
    return test_acc


if __name__ == '__main__':
    # GPU is available? If so, we use it.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}!")
        torch.backends.cudnn
        torch.backends.cudnn.enabled = False 
        benchmark = True
    else:
        print("No GPU detected. Will use CPU for training.")
    hidden_dict = {100: 1200, 200: 5808, 300: 13068, 400: 23232, 500: 36300}
    parc = 100
    for ref in ['none', 'pca']:
        print(f'Using Schaefer{parc} parcellation')
        # Navigate tree and get raw correlation FC matrices
        print("Importing all correlation matrices...", end=" ")
        all_FC, nSubj, nFCs = get_data(parc)
        if ref == 'pca':
            all_FC = pca_recon(all_FC, 0.012)
        print("All FCs successfully loaded!\n")
        replicates = np.arange(1, 21)
        all_acc, all_loss = {}, {}
        # Prepare train, validation, and test data for NN
        print("Preparing data for CNN...", end=" ")
        train_loader, val_loader, test_loader = prepare_data(
            all_FC, nSubj, nFCs)
        print("done!\n")
        # Max epochs of training, early stopping threshold, learning rate
        max_epochs, n_epochs_stop, lr = 200, 5, 0.001
        # Loop over iterations of the model
        for rep in replicates:
            model, loss_fn, opt, history = build_model(
                lr, hidden_dict[parc])
            print(f"{ref}: Training model {rep} of {replicates[-1]}...")
            model, history = train_model(model, opt, loss_fn, train_loader,
                                         val_loader, max_epochs, n_epochs_stop,
                                         history)
            accuracy = test_model(model, test_loader)
            all_acc[rep] = accuracy
            all_loss[rep] = min(history['val_loss'])
            print(
                f'Model {rep} - Accuracy: {accuracy}; Loss: {all_loss[rep]}')
        # Write to dataframe and to csv
        filename = f'../results/tasks/CNN_schaefer{parc}_{ref}.csv'
        results = pd.DataFrame.from_dict(
            all_acc, orient='index', columns=['Accuracy'])
        results["Loss"] = pd.Series(all_loss)
        results.to_csv(filename)

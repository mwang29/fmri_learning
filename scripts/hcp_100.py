import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
import sklearn.datasets
import sklearn.decomposition
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(18252, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (3,3)))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), (3,3)))
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

def get_data():
    '''
    Navigates through file tree and extracts FCs with optional reconstruction
    '''
    fname = '../data/100_unrelated.csv'
    yeo_order = list(sio.loadmat("../data/yeo_RS7_N374.mat", squeeze_me=True, struct_as_record=False)['yeoOrder']-1)
    subjectids = np.loadtxt(fname, dtype=np.int)
    nSubj = len(subjectids)
    tasks = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR',
             'rfMRI_REST2_RL', 'tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL',
             'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL', 'tfMRI_LANGUAGE_LR',
             'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
             'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR',
             'tfMRI_SOCIAL_RL', 'tfMRI_WM_LR', 'tfMRI_WM_RL']

    M = {}
    for task in tasks:
        masterFC_dir = '../data/results_SIFT2'
        restingstatename = 'fMRI/' + task + '/FC/FC_glasser_subc_GS_bp_z.mat'
        task_matrices = []
        for subject in subjectids:
            filename = masterFC_dir + '/' + \
                str(subject) + '/' + restingstatename
            mat = sio.loadmat(filename, squeeze_me=True,
                              struct_as_record=False)
            A_orig = mat['FC']
            A_orig = A_orig[np.ix_(yeo_order, yeo_order)]
            np.fill_diagonal(A_orig, 0)
            task_matrices.append(A_orig)
        M[task] = np.array(task_matrices)
    task_dict = {}
    task_dict["Rest"] = np.concatenate((M['rfMRI_REST1_LR'],M['rfMRI_REST1_RL']))
    task_dict["Emotion"] = np.concatenate((M['tfMRI_EMOTION_LR'],M['tfMRI_EMOTION_RL']))
    task_dict["Gambling"] = np.concatenate((M['tfMRI_GAMBLING_LR'], M['tfMRI_GAMBLING_RL']))
    task_dict["Language"] = np.concatenate((M['tfMRI_LANGUAGE_LR'], M['tfMRI_LANGUAGE_RL']))
    task_dict["Motor"] = np.concatenate((M['tfMRI_MOTOR_LR'], M['tfMRI_MOTOR_RL']))
    task_dict["Relational"] = np.concatenate(( M['tfMRI_RELATIONAL_LR'], M['tfMRI_RELATIONAL_RL']))
    task_dict["Social"] = np.concatenate((M['tfMRI_SOCIAL_LR'], M['tfMRI_SOCIAL_RL']))
    task_dict["WM"] = np.concatenate((M['tfMRI_WM_LR'], M['tfMRI_WM_RL']))
    del M
    all_FC = np.concatenate(tuple(task_dict.values()))
    del task_dict
    return all_FC, nSubj


def pca_recon(FC, pctComp=None):
    '''
    Reconstructs FC based on number of principle components
    '''
    if pctComp is None:
        return FC
    FC = np.reshape(FC, (FC.shape[0], -1))
    nComp = int(FC.shape[0] * pctComp)
    mu = np.mean(FC, axis=0)
    pca_rest = sklearn.decomposition.PCA()
    pca_rest.fit(FC)

    SCORES = pca_rest.transform(FC)[:, :nComp]
    COEFFS = pca_rest.components_[:nComp, :]
    FC_recon = np.dot(SCORES, COEFFS)
    FC_recon += mu
    FC_recon = np.reshape(FC_recon, (FC.shape[0], 374, 374))
    return FC_recon


def prepare_data(all_FC, nSubj):
    '''
    Prepares labels and train, val and test data from raw data
    '''
    labels = torch.tensor(np.repeat(np.arange(0,8),nSubj*2), dtype=torch.long)
    indices = np.random.permutation(labels.shape[0])
    train_idx = indices[:int(0.6*labels.shape[0])]
    val_idx = indices[int(0.6*labels.shape[0]):int(0.8*labels.shape[0])]
    test_idx = indices[int(0.8*labels.shape[0]):]
    train_mean = np.mean(all_FC[train_idx])
    train_std = np.std(all_FC[train_idx])
    train_data = torch.FloatTensor((all_FC[train_idx] - train_mean) / train_std)
    val_data = torch.FloatTensor((all_FC[val_idx] - train_mean) / train_std)
    test_data = torch.FloatTensor((all_FC[test_idx] - train_mean) / train_std)
    
    train_data = train_data.view(train_data.shape[0], -1, train_data.shape[1], train_data.shape[2])
    val_data = val_data.view(val_data.shape[0], -1, val_data.shape[1], val_data.shape[2])
    test_data = test_data.view(test_data.shape[0], -1, test_data.shape[1], test_data.shape[2])

    train_dataset = data.TensorDataset(train_data,labels[train_idx]) # create your datset
    val_dataset = data.TensorDataset(val_data,labels[val_idx]) # create your datset
    test_dataset = data.TensorDataset(test_data,labels[test_idx]) # create your datset

    train_loader = data.DataLoader(train_dataset, batch_size=80) # create your dataloader
    val_loader = data.DataLoader(val_dataset, batch_size=80) # create your dataloader
    test_loader = data.DataLoader(test_dataset, batch_size=80) # create your dataloader


    return train_loader, val_loader, test_loader


def build_model(lr):
    '''
    Given layer sizes and learning rate, builds model.
    Can change NN architecture here directly in nn.Sequential
    '''
    model = Net()
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
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            if use_cuda:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)



            opt.zero_grad()
            output = model(local_batch)
            loss = loss_fn(output, local_labels)
            loss.backward()
            opt.step()

            train_loss += loss.data.item() * local_batch.size(0)
            num_train_correct  += (torch.max(output, 1)[1] == local_labels).sum().item()
            num_train_examples += local_batch.shape[0]

            train_acc = num_train_correct / num_train_examples
            train_loss  = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in val_loader:
                # Transfer to GPU
                if use_cuda:
                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model(local_batch)
                loss = loss_fn(output, local_labels)

                val_loss += loss.data.item() * local_batch.size(0)
                num_val_correct  += (torch.max(output, 1)[1] == local_labels).sum().item()
                num_val_examples += local_batch.shape[0]

            val_acc  = num_val_correct / num_val_examples
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
        print("GPU detected. Will use GPU for training!")
        torch.backends.cudnn
        benchmark = True
    else:
        print("No GPU detected. Will use CPU for training.")
    pctComp = list(np.arange(0.025, 1, step=0.025))
    all_acc, all_loss = {}, {}
    all_FC, nSubj = get_data()
    for comp in pctComp:
        # Get data from file tree
        temp_FC = pca_recon(all_FC, pctComp=comp)
        print(f"Reconstructed at {int(comp*100)}% components")
        # Prepare train, validation, and test data for NN
        train_loader, val_loader, test_loader = prepare_data(temp_FC, nSubj)
        del temp_FC
        # Maximum epochs of training, early stopping threshold, and learning rate
        max_epochs, n_epochs_stop, lr = 200, 5, 0.001
        # Build model accordingly
        model, loss_fn, opt, history = build_model(lr)
        print("Built model. Now training...")
        model, history = train_model(model, opt, loss_fn, train_loader, val_loader,
                            max_epochs, n_epochs_stop, history)
        accuracy = test_model(model, test_loader)
        all_acc[comp] = accuracy
        all_loss[comp] = min(history['val_loss'])
        del model, train_loader, val_loader, test_loader
        print(f'Test accuracy of model is {accuracy}')
    acc_filename = f'acc_{max_epochs}_{lr}.csv'
    loss_filename = f'loss_{max_epochs}_{lr}.csv'
    with open(f'../results/{acc_filename}', 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in all_acc.items():
            writer.writerow([key, value])
    with open(f'../results/{loss_filename}', 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in all_loss.items():
            writer.writerow([key, value])



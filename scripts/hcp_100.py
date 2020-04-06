import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler


def get_data():
    fname = '../data/100_unrelated.csv'
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
            A_orig = (A_orig + A_orig.T) / 2
            np.fill_diagonal(A_orig, 0)
            task_matrices.append(A_orig)
        # This is the 100 by 374 by 374 matrices of 100 unrelated subjects
        task_matrices = np.array(task_matrices).reshape(len(subjectids), -1)
        M[task] = task_matrices

    all_FC = np.hstack((M['rfMRI_REST1_LR'], M['rfMRI_REST1_RL'],
                        M['tfMRI_EMOTION_LR'], M['tfMRI_EMOTION_RL'],
                        M['tfMRI_GAMBLING_LR'], M['tfMRI_GAMBLING_RL'],
                        M['tfMRI_LANGUAGE_LR'], M['tfMRI_LANGUAGE_RL'],
                        M['tfMRI_MOTOR_LR'], M['tfMRI_MOTOR_RL'],
                        M['tfMRI_RELATIONAL_LR'], M['tfMRI_RELATIONAL_RL'],
                        M['tfMRI_SOCIAL_LR'], M['tfMRI_SOCIAL_RL'],
                        M['tfMRI_WM_LR'], M['tfMRI_WM_RL'])).reshape(-1, M['tfMRI_WM_LR'].shape[1])
    return all_FC, nSubj


def prepare_data(all_FC, nSubj):
    labels = torch.tensor(np.repeat(np.arange(0, 8), nSubj * 2))
    indices = np.random.permutation(labels.shape[0])
    train_idx = indices[:int(0.6 * labels.shape[0])]
    val_idx = indices[int(0.6 * labels.shape[0]):int(0.8 * labels.shape[0])]
    test_idx = indices[int(0.8 * labels.shape[0]):]
    std_scale = StandardScaler().fit(all_FC[train_idx, :])
    all_FC[train_idx, :] = std_scale.transform(all_FC[train_idx, :])
    all_FC[val_idx, :] = std_scale.transform(all_FC[val_idx, :])
    all_FC[test_idx, :] = std_scale.transform(all_FC[test_idx, :])
    all_FC = torch.Tensor(all_FC)

    train_data = []
    for i in train_idx:
        train_data.append([all_FC[i], labels[i]])

    val_data = []
    for i in val_idx:
        val_data.append([all_FC[i], labels[i]])

    test_data = []
    for i in test_idx:
        test_data.append([all_FC[i], labels[i]])

    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=100)
    val_loader = torch.utils.data.DataLoader(
        val_data, shuffle=True, batch_size=100)
    test_loader = torch.utils.data.DataLoader(
        test_data, shuffle=True, batch_size=100)

    return all_FC, train_loader, val_loader, test_loader


def build_model(sizes, lr):
    model = nn.Sequential(nn.Linear(sizes[0], sizes[1]),
                          nn.BatchNorm1d(sizes[1]),
                          nn.ReLU(),
                          nn.Dropout(),
                          nn.Linear(sizes[1], sizes[2]),
                          nn.BatchNorm1d(sizes[2]),
                          nn.ReLU(),
                          nn.Linear(sizes[2], sizes[3]))
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    return model, loss_fn, opt, history


def train_model(model, opt, loss_fn, train_loader, val_loader,
                test_loader, max_epochs, n_epochs_stop, history):
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
    return model


def test_model(model, test_loader):
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
        cudnn.benchmark = True
    else:
        print("No GPU detected. Will use CPU for training.")

    all_FC, nSubj = get_data()
    all_FC, train_loader, val_loader, test_loader = prepare_data(all_FC, nSubj)
    sizes = [all_FC.shape[1], 2048, 128, 8]
    max_epochs, n_epochs_stop, lr = 30, 5, 0.01
    model, loss_fn, opt, history = build_model(sizes, lr)
    model = train_model(model, opt, loss_fn, train_loader, val_loader,
                        max_epochs, n_epochs_stop, history)
    accuracy = test_model(model, test_loader)

# Loop over epochs

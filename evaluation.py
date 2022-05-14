import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import accuracy_score


n_epochs = 4
# load the loss function
criterion = nn.BCELoss()
# load the optimizer

def eval_model(model, dataloader):
    """
    :return:
        Y_pred: prediction of model on the dataloder.
            Should be an 2D numpy float array where the second dimension has length 2.
        Y_test: truth labels. Should be an numpy array of ints
    TODO:
        evaluate the model using on the data in the dataloder.
        Add all the prediction and truth to the corresponding list
        Convert Y_pred and Y_test to numpy arrays (of shape (n_data_points, 2))
    """
    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    for s0, s1, y in dataloader:
        # your code here
        pred_result = model(s0, s1).detach().to('cpu')
        y_score = torch.cat((y_score,  pred_result), dim=0)
        pred_result = pred_result > 0.8
        y_pred = torch.cat((y_pred,  pred_result), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    acc = accuracy_score(y_pred, y_true)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return acc, p, r, f, roc_auc


def train_model(model, train_dataloader, optimizer, n_epoch=n_epochs, criterion=criterion, to_print=False):
    import torch.optim as optim
    """
    :param model: A CNN model
    :param train_dataloader: the DataLoader of the training data
    :param n_epoch: number of epochs to train
    :return:
        model: trained model
    """
    model.train() # prep model for training
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for s0, s1, y in train_dataloader:
            # your code here
            optimizer.zero_grad()
            y_hat = model(s0, s1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())
        if to_print:
            print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
class SentenceDataset(Dataset):

    def __init__(self, embedding_model, file_name = 'cnn_data.csv'):
        data = pd.read_csv(file_name, delimiter=',')
        X = pd.DataFrame()
        X[0] = data['0']
        X[1] = data['1']
        self.X = X.values
        Y = pd.DataFrame()
        Y[0] = data['2']
        self.Y = Y.values
        self.embedding_dictionary = embedding_model.key_to_index
        self.padding_index = len(self.embedding_dictionary)

    def __len__(self):

        """
        TODO: Return the number of samples (i.e. admissions).
        """

        # your code here
        return len(self.X)

    def __getitem__(self, index):

        """
        TODO: Generate one sample of data.

        Hint: convert text to indices using to_index();
        """

        text = self.X[index]
        label = self.Y[index]
        # split into [word]
        text = [i.split(' ') for i in text]
        text = [i + max(5 - len(i), 0) * ['placeeeeeeeeeeeholder'] for i in text]
        # word to index in vocab
        text = [[self.embedding_dictionary[j] if j in self.embedding_dictionary \
                 else self.padding_index for j in i ] for i in text]

        # return text as long tensor, labels as float tensor;
        return torch.tensor(text[0], dtype=torch.long), torch.tensor(text[1], dtype=torch.long), torch.tensor(label, dtype=torch.float)

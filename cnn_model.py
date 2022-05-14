import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class SimilarityCNN(nn.Module):

    def __init__(self, embeding_model=None):
        super().__init__()
        Co = 32
        Ks = [3, 4, 5]
        embedding_weights = embeding_model.vectors
        embedding_weights = np.append(embedding_weights, np.array([np.zeros(embedding_weights[0].shape)]), axis=0)
        self.embeding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))
        D = self.embeding.embedding_dim
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (K, D)) for K in Ks])
        self.fc1 = nn.Linear(3 * 32 * 2, 1)

    def forward(self, s0, s1):
        embeded_1 = self.embeding(s0).unsqueeze(1)
        conv_1 = [F.relu(conv(embeded_1)).squeeze(3) for conv in self.convs]
        pool_1 = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in conv_1]
        pool_1 = torch.cat(pool_1, 1)



        embeded_2 = self.embeding(s1).unsqueeze(1)
        conv_2 = [F.relu(conv(embeded_2)).squeeze(3) for conv in self.convs]
        pool_2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_2]
        pool_2 = torch.cat(pool_2, 1)
        final_vector = torch.concat((pool_1, pool_2), dim=1)
        output = self.fc1(final_vector)
        return nn.Sigmoid()(output)

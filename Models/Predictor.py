from torch import nn
import torch

from Models.Classifier import Classifier


class Predictor(nn.Module):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list):
        super(Predictor, self).__init__()
        self.size_subsequent = size_subsequent
        self.classifier = classifier
        self.snippet_list = snippet_list
        self.dim = dim
        self.num_layers = 1
        self.hidden_dim = 128
        self.gru = nn.GRU(input_size=self.dim * 2,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.relu = nn.ReLU()
        self.fs1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.relu1 = nn.ReLU()
        self.fs2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.relu2 = nn.ReLU()
        self.fs3 = nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8)
        self.relu3 = nn.ReLU()
        self.last = nn.Linear(self.hidden_dim // 8 + dim, dim)
        self.relu_last = nn.ReLU()

    def forward(self, x, h):
        snippet = self.classifier(x)
        snip = self.snippet_tensor(snippet)
        last = snip[:, :, -1]
        snip = snip[:, :, :-1]
        x = torch.cat((x, snip), dim=1)
        x = x.transpose(1, 2)
        x, h = self.gru(x)
        #

        x = x[:, -1, :]
        #x = nn.LeakyReLU()(x)
        x = self.fs1(x)
        x = nn.LeakyReLU()(x)
        x = self.fs2(x)
        x = nn.LeakyReLU()(x)
        x = self.fs3(x)
        x = nn.LeakyReLU()(x)
        x = torch.cat((x, last), dim=1)
        x = self.last(x)
        x = nn.LeakyReLU()(x)
        return x, snippet.argmax(dim=1)

    def snippet_tensor(self, snippet):
        arr = []
        snip = snippet.argmax(dim=1)
        for i in snip:
            arr.append(self.get_snippet(i.tolist()))
        return torch.cat(arr, dim=0)

    def get_snippet(self, numbers):
        return_arr = []
        for ids, number in enumerate(numbers):
            return_arr.append(self.snippet_list[ids][number])

        return torch.tensor([return_arr])

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers,
                            batch_size,
                            self.hidden_dim).zero_().to(torch.device('cpu'))
        return hidden

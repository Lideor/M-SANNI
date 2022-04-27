from torch import nn
import torch

from Models.Predictors.Predictor import Predictor
from Models.Classifier import Classifier


class OnlyCnns(Predictor):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list,
                 device,
                 hidden_dim=128,
                 num_layers=1,
                 ):
        super(OnlyCnns, self).__init__(
            size_subsequent,
            count_snippet,
            dim,
            classifier,
            snippet_list,
            device,
            hidden_dim,
            num_layers,
        )
        self.cnns1 = nn.ModuleList([nn.Conv1d(
            in_channels=2,
            out_channels=128,
            padding=2,
            kernel_size=(5,)) for i in range(dim)])
        self.cnns2 = nn.ModuleList([nn.Conv1d(
            in_channels=128,
            out_channels=64,
            padding=2,
            kernel_size=(5,)) for i in range(dim)])
        self.cnns3 = nn.ModuleList([nn.Conv1d(
            in_channels=64,
            out_channels=32,
            padding=2,
            kernel_size=(5,)) for i in range(dim)])
        self.last_cnn = 32
        # for i in range(dim):
        #     conv = nn.Conv1d(
        #         in_channels=2,
        #         out_channels=1,
        #         padding=2,
        #         kernel_size=(5,)).to(self.device)
        #     self.cnns.append(conv)
        self.gru = nn.GRU(input_size=self.dim * 32,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)

    def augmentation(self, x):
        with torch.no_grad():
            snippet = self.classifier(x).argmax(dim=1).cpu()
        snip = self.snippet_tensor(snippet)
        last = snip[:, :, -1]
        snip = snip[:, :, :-1]
        result_x = torch.zeros(size=(x.shape[0], x.shape[1] * 32, x.shape[2]), device=self.device)
        for i, cnn in enumerate(self.cnns3):
            input_cnn = torch.cat((x[:, i:i + 1, :], snip[:, i:i + 1, :]), dim=1)
            input_cnn = self.cnns1[i](input_cnn)
            input_cnn = self.cnns2[i](input_cnn)
            result = cnn(input_cnn)
            result_x[:, i:i + result.shape[1], :] = result
        return result_x, last

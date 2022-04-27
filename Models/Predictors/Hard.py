from torch import nn
import torch

from Models.Predictors.OnlyCnns import OnlyCnns
from Models.Classifier import Classifier


class Hard(OnlyCnns):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list,
                 device,
                 hidden_dim=128,
                 num_layers=1,
                 fc_gru=5,
                 ):
        super(Hard, self).__init__(
            size_subsequent,
            count_snippet,
            dim,
            classifier,
            snippet_list,
            device,
            hidden_dim,
            num_layers,
        )
        self.gru_dim = nn.ModuleList([nn.GRU(input_size=self.last_cnn,
                                             hidden_size=self.size_subsequent-1,
                                             # num_layers=1,
                                             batch_first=True) for i in range(dim)])
        self.gru = nn.GRU(input_size=self.dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)

    def augmentation(self, x):
        with torch.no_grad():
            snippet = self.classifier(x).argmax(dim=1).cpu()
        snip = self.snippet_tensor(snippet)
        last = snip[:, :, -1]
        snip = snip[:, :, :-1]
        result_x = torch.zeros(size=x.shape, device=self.device)
        for i, cnn in enumerate(self.cnns3):
            input_cnn = torch.cat((x[:, i:i + 1, :], snip[:, i:i + 1, :]), dim=1)
            input_cnn = self.cnns1[i](input_cnn)
            input_cnn = self.cnns2[i](input_cnn)
            result = cnn(input_cnn)
            result, _ = self.gru_dim[i](result.transpose(1, 2))
            result_x[:, i:i + 1, :] = result[:,None, -1, :]
        return result_x, last

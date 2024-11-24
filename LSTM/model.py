import torch
import torch.nn as nn
from entmax import EntmaxBisect
import numpy as np

np.random.seed(42)


class LSTMAllocationModelWithAttention(nn.Module):
    def __init__(self, input_size=592, hidden_sizes=[32, 64, 148], dropout_rate=0.5, recent_days=7, entmax_alpha=1.5, decay_weight=2):
        super(LSTMAllocationModelWithAttention, self).__init__()

        self.decay_weight = decay_weight
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.recent_days = recent_days
        self.entmax_alpha = entmax_alpha

        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True))

        self.dropout = nn.Dropout(p=dropout_rate)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes])

        self.attention_weights = nn.Linear(hidden_sizes[-1], 1, bias=False)

        self.entmax = EntmaxBisect(dim=1, alpha=entmax_alpha)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Custom weight initialization for all layers.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:  
                    nn.init.xavier_normal_(param)  
                elif 'attention_weights' in name: 
                    nn.init.xavier_uniform_(param) 
            elif 'bias' in name:
                nn.init.zeros_(param)  


    def forward(self, x):

        for i, lstm in enumerate(self.lstm_layers):
            h0 = torch.zeros(1, x.size(0), self.hidden_sizes[i], device=x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_sizes[i], device=x.device)

            x, _ = lstm(x, (h0, c0))
            x = x.permute(0, 2, 1)

            if x.shape[0] != 1:
                x = self.batch_norms[i](x)

            x = x.permute(0, 2, 1)
            x = self.dropout(x)


        attn_scores = self.attention_weights(x).squeeze(-1)  

 r
        recent_mask = torch.ones_like(attn_scores)
        recent_mask[:, -self.recent_days:] *= self.decay_weight  
        attn_scores = attn_scores * recent_mask 

        attn_weights = torch.softmax(attn_scores, dim=1)  
        x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1) 

        allocations = self.entmax(x)

        return allocations

import torch
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device):
        super().__init__()
        self.device = device
        self.input_size = input_size # embedding size
        self.hidden_size = hidden_size # hidden size
        self.num_layers = num_layers # number of layers
        self.output_size = output_size # output size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size # batch size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):  # input_seq: (batch_size, seq_len, input_size)
        batch_size, seq_len = input_seq[0], input_seq[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq) # output(5, 30, 64)

        pred = self.linear(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


# lstm = LSTM(input_size=5, hidden_size=64, num_layers=2, output_size=1, batch_size=5, device=torch.device('cuda'))
# lstm.to(lstm.device)

# input_seq = torch.randn(5, 30, 5).to(lstm.device) #[Batch_size, seq_len, embedding_size]

# output = lstm(input_seq)
# print(output.shape)





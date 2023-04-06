import torch
from torch.nn import functional
from AbstractRNNCell import AbstractRNNCell

class MyLSTMCell(AbstractRNNCell):

    def __init__(self, input_size, hidden_size, output_size, device):
        """
        Create an LSTM operating on inputs of feature dimension input_size,
        outputting outputs of feature dimension output_size and maintaining a
        hidden state of size hidden_size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        #####Insert your code here for subtask 2a#####
        self.has_hidden_state = False
        self.device = device

    def get_rnn_type(self):
        return "LSTM"

    def forward(self, x : torch.tensor, reset_hidden_state : bool = True):
        if(len(x.shape) < 1): #batch sz 1, seq len 1, 1-d features
            x = x[None][None, None]
        elif(len(x.shape) < 2): #seq len 1, 1-d features
            x = x[:, None, None]
        elif(len(x.shape) < 3): #1-d features
            x = x[:, :, None]
        batch_size = x.shape[0]
        sequence_len = x.shape[1]

        if(reset_hidden_state or not self.has_hidden_state):
            self.hidden_state = torch.zeros((self.hidden_size, batch_size), 
                device=self.device)
            self.cell_state = torch.zeros((self.hidden_size, batch_size), 
                device=self.device)            
            self.has_hidden_state = True
        outputs = torch.zeros((x.shape[0], x.shape[1], self.output_size), 
            device=self.device)
        
        for t in range(sequence_len):
            curr_x = x[:, t, :]
            #####Insert your code here for subtask 2b#####

        return torch.squeeze(outputs)

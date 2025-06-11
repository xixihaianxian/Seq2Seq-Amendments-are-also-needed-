import torch
from torch import nn
from torch.utils import data

class Encoder(nn.Module):
    def __init__(self,words,embedding_dim=100,padding_idx=0,input_size=100,hidden_size=256,num_layers=2,dropout_rate=0.2,bidirectional=True):
        super().__init__()
        self.words=words
        self.embedding_dim=embedding_dim
        self.padding_idx=padding_idx
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.embedding=nn.Embedding(num_embeddings=len(self.words),embedding_dim=self.embedding_dim,padding_idx=self.padding_idx)
        self.lstm=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,bidirectional=self.bidirectional)
        self.dropout=nn.Dropout(p=dropout_rate)
        self.linear=nn.Linear(in_features=self.hidden_size*(2 if self.bidirectional else hidden_size),out_features=len(words))

    def forward(self,input:torch.Tensor):
        input=self.dropout(self.embedding(input))
        # input=nn.utils.rnn.pack_padded_sequence(input,input_length,batch_first=True)
        output,(hidden,cell)=self.lstm(input)
        # output=nn.utils.rnn.pad_packed_sequence(sequence=output,batch_first=True)
        return output,hidden,cell
        #TODO output(batch_size,sequence_length,hidden_size*2) hidden(batch_size,hidden_layer*2,hidden_size) cell和hidden相同

class Decorder(nn.Module):

    def __init__(self,words,embedding_dim=100,padding_idx=0,input_size=100,hidden_size=256,num_layer=2,bidirectional=True,dropout_rate=0.2):
        super().__init__()
        self.words=words
        self.embedding_dim=embedding_dim
        self.padding_idx=padding_idx
        self.embedding=nn.Embedding(num_embeddings=len(words),embedding_dim=self.embedding_dim,padding_idx=self.padding_idx)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.bidirectional=bidirectional
        self.dropout_rate=dropout_rate
        self.lstm=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layer,batch_first=True,bidirectional=self.bidirectional)
        self.dropout=nn.Dropout(p=dropout_rate)
        self.linear=nn.Linear(in_features=self.hidden_size*(2 if self.bidirectional else 1),out_features=len(words))

    def forward(self,input,hidden,cell):
        input=self.dropout(self.embedding(input))
        output,(hidden,cell)=self.lstm(input,(hidden,cell))
        output=self.linear(output)
        return output,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,people,robot):
        encoder_output,encoder_hidden,encoder_cell=self.encoder(people)
        robot=robot[:,:-1]
        decoder_output,decoder_hidden,decoder_cell=self.decoder(robot,encoder_hidden,encoder_cell)
        return decoder_output
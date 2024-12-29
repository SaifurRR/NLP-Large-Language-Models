<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch.nn as nn
import torch


# We&#39;ve started building the `CharacterLSTM` class for our character-based LSTM. 
# 
# Let&#39;s first initialize the LSTM components in the `init` method as follows:
# 1. Create an embedding layer named `self.embedding` that inputs neurons equal to the vocabulary size (which is `51` for our character-based vocabulary) and an embedding dimension of `64`
# 2. Create an LSTM layer named `self.lstm` that takes an input size equal to the embedding dimension and a hidden size of `128`.
#     - We need to ensure we also specify `batch_first=True`
# 4. Create the linear output layer named `self.linear` that takes an input equal to the hidden size of the LSTM layer and outputs neurons equal to the vocabulary size
# 

# In[3]:


import torch.nn as nn
torch.manual_seed(1) # set random seed 

class CharacterLSTM(nn.Module):
    def __init__(self):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(51, embedding_dim = 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(128, 51)

    def init_state(self, batch_size):
        hidden = torch.zeros(1, batch_size, 24)
        cell = torch.zeros(1, batch_size, 24)
        return hidden, cell


# Next, let&#39;s build the forward method that takes in the input (`x`) and the hidden and cell states (`states`) with the following:
# 
# 1. Pass the input `x` through the embedding layer
# 2. Pass the embedding output along with the previous states to the LSTM layer and return the output and the updated states
# 3. Pass the LSTM output to the linear layer
# 4. Reshape the linear layer output
# 5. Lastly, return the reshaped output and the updated states
# 

# In[4]:


import torch.nn as nn
torch.manual_seed(1) # set random seed

class CharacterLSTM(nn.Module):
    def __init__(self):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(51, embedding_dim = 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(128, 51)

    def forward(self, x, states):
        x = self.embedding(x)
        out, states = self.lstm(x,states)
        out = self.linear(out)
        out = out.reshape(-1,out.size(2))
        return out, states

    def init_state(self, batch_size):
        hidden = torch.zeros(1, batch_size, 128)
        cell = torch.zeros(1, batch_size, 128)
        return hidden, cell


# Lastly, let&#39;s create an instance of the LSTM class and save it to the variable `lstm_model`.
# 

# In[7]:


lstm_model = CharacterLSTM()
lstm_model

<script type="text/javascript" src="/relay.js"></script></body></html>

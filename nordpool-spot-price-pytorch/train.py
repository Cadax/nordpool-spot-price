from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models import CNN, CNNLSTM
import pandas as pd
from utils import SpotPriceDataset

df = pd.read_pickle('dataset.pickle')
sequence_len, target = 24, 0
dataset = SpotPriceDataset(df,sequence_len,target)
dataset.prepare_data()
dataloader = DataLoader(dataset, batch_size=2**4, num_workers=0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

model = CNNLSTM(dataset.variables,sequence_len)
model.to(device)

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 5
for epoch  in range(epochs):
    total_loss = 0
    for batch in dataloader:
        input_seq, target_seq = batch
        print(input_seq.shape,target_seq.shape)
        input_seq = input_seq.to(device)
        output = model(input_seq)
        loss = criterion(output, target_seq.view(-1).long())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    
    print(f"Epoch : {epoch}/{epochs} /r/n")
    print(f"Training Loss : {total_loss:.3f} /r/n")
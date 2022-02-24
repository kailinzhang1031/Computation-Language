## 3. POS Tagging Based On Recurrent Neural Network - LSTM

### 3.1 Weight Instantiation

```python
def init_weights(model):
    for param in model.parameters():
        torch.nn.init.uniform_(param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE)
```

## 2. Model


### 2.1 Code
```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)
        init_weights(self)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        outputs = self.output(hidden)
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs
```

### 2.2 Check Point

The forward function goes as :
- embedding
- packing
- lstm
- unpacking
- output
- softmax

#### 2.2.1 Preliminary

- Packed Sequence

RNN network always use **packed series**.

The **packed series** is beneficial to reduce the computations in every batch.

Click here for detailed description: [why do we pack the sequences in pytorch](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch)

- LSTM

```h_n```: tensor of shape (D * \text{num\_layers}, N, H_{out})(D∗num_layers,N,H 
out) containing the final hidden state for each element in the batch.

```c_n```: tensor of shape (D * \text{num\_layers}, N, H_{cell})(D∗num_layers,N,H 
cell) containing the final cell state for each element in the batch.

#### 2.2.2 Practice

The code of model is the same as text classification, except for the following context:

Inverse to ```pack_padded_sequence```, ```pad_packed_sequence```  unpacks the **packed series**,return them to
several padded series.

```python
hidden, _ = pad_packed_sequence(hidden, batch_first=True)
```

Only hidden layer(hc) in the last state is used in text classification, while in POS tagging,
we need to use hidden layers of **all states of series**.

### 3.3 Training And Testing

```python
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    log_probs_list = []
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets, mask = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        log_probs_list.append(log_probs)
        loss = nll_loss(log_probs[mask], targets[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(log_probs_list)
    print(f"Loss: {total_loss:.2f}")
```

```python
acc = 0
total = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets, mask = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=-1) == targets)[mask].sum().item()
        total += mask.sum().item()

print(f"Acc: {acc / total:.2f}")
```


We need to use mask to ensure that loss of effective tokens are calculated and right predict result
and the wholes tokens are taken into account.

```python
        loss = nll_loss(log_probs[mask], targets[mask])
        acc += (output.argmax(dim=-1) == targets)[mask].sum().item()
```


## 3.4 Check Point

To illustrate how LSTM works, we will visualize the whole process of experiment.

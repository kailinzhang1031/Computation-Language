# 4.6 Motion Classification


Here we introduce motion classification, a practical case study in NLP.

This checkpoint is organized by:
- [0.Data Exploration]()
- [1.Data Preprocessing]()
    - [1.1 Vocabulary Mapping ]()
    - [1.2 Data Loading]()
    - [1.3 Dataset Construction]()
    - [1.4 Data Transformation]()
- [2.Multilayer Perceptron Combined With Word Vector]()
- [3.Training and Testing Based On Multilayer Perceptron]()
- [4.1 Motion Classification Based On CNN]()
- [4.2 Motion Classification Based On RNN]()
- [4.3 Motion Classification Based On Transformer]()

Prerequisite relevant learning:
- [1.Vocabulary Mapping]()
- [2.Word Vector Layer]()

## 0. Data Exploration

### 0.1 Introduction

For detailed description, please click here: [Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/)

### 0.2 Quantified Features
- Size: 10662(5331+5331)
- Samples:
    | Null(Index) | 0(Sentence) | 1(Label) |
    | :--: | :--: | :--: |
    | 0 | simplistic , silly and tedious . | neg |
    | 1 | it's so laddish and juvenile , only teenage boys could possibly find it funny . | neg |
    | 5330 | enigma is well-made , but it's just too dry and too placid . | neg |
    | 5331 | the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . | pos |
    | 5332 | the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth . | pos |
    | 10661 | provides a porthole into that noble , trembling incoherence that defines us all . | pos |

 ## 1. Data Preprocessing

### 1.1.1 Vocabulary Mapping

Convert token to integers, which >= 0 and < size of vocabulary.

```python
from collections import defaultdict, Counter

class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]
```

### 1.1.2 Data Loading

```python
import torch
from vocab import Vocab

def load_sentence_polarity():
    from nltk.corpus import sentence_polarity

    vocab = Vocab.build(sentence_polarity.sents())

    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data, test_data, vocab
```


### 1.1.3 Dataset Construction

```python
class BowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
```

### 1.1.4 Data Transformation

```python
def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets
```

```inputs = pad_sequence(inputs, batch_first=True)```:

Padding samples in a batch to the same length.

### 1.2 Experiment

- Vocabulary
    - length of 21402
    - vocab.idx_to_token: 
        ```python
        ['<unk>', 'simplistic', ',', 'silly', 'and', 'tedious', '.', "it's", 'so', 'laddish', 'juvenile', 'only', 'teenage', 'boys', 'could', 'possibly', 'find', 'it', 'funny',
        ...
         "[tsai's]", 'mazel', 'tov', 'depleted', 'piscopo', 'chaykin', 'headly', 'porthole']
        ```
    - vocab.token_to_idx:
        ```python
        {'<unk>': 0, 'simplistic': 1, ',': 2, 'silly': 3, 'and': 4, 'tedious': 5, '.': 6,
        ...
         "[tsai's]": 21394, 'mazel': 21395, 'tov': 21396, 'depleted': 21397, 'piscopo': 21398, 'chaykin': 21399, 'headly': 21400, 'porthole': 21401}
        ```
- Train Data:
    - length of 8000 = 4000 * 2
    - train_data[0]
      ```python
      ([23, 2444, 61, 9851, 76, 308, 23, 1664, 14509, 496, 219, 14510, 219, 4, 27, 175, 363, 76, 29, 32, 5884, 201, 7984, 73, 5354, 4219, 2, 14511, 1204, 2701, 25, 2184, 14512, 6], 0)
      ```
    - train_data[7999]
      ```python
      ([547, 2003, 2101, 371, 76, 98, 6], 1)
      ```
- Test Data:
    - length of 2662 = 1331 * 2
    - test_data[0]
      ```python
      ([2430, 105, 3145, 4, 14750, 442, 19982, 746, 19983, 2, 162, 15638, 435, 23, 1364, 438, 8688, 2, 111, 12121, 4376, 8666, 31, 63, 6778, 4328, 76, 4376, 8666, 105, 1468, 1975, 5216, 6], 0)
      ```
    - test_data[2662]
      ```python
      ([2136, 61, 4008, 2, 51, 7, 388, 782, 4148, 4, 782, 14508, 6], 1)
      ```
- Train Data loader
    - batch_size = 32
    - length = 250
    - sample in a batch(without shuffling):
      ```python
      tensor([[   23,  2444,    61,  ...,     0,     0,     0],
        [   23,  4733,  5842,  ...,     6,     0,     0],
        [ 2745,    51, 14516,  ...,     0,     0,     0],
        ...,
        [   17,  5127,    27,  ...,     0,     0,     0],
        [ 8532,    76,  1364,  ...,     0,     0,     0],
        [   32, 14546,   228,  ...,     0,     0,     0]])  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])
      ```
      Tensor of torch.Size([32, 41])

- Test Data Loader
    - batch_size = 1
    - length = 34
    - sample in a batch(without shuffling):
      ```python
      tensor([[ 2430,   105,  3145,     4, 14750,   442, 19982,   746, 19983,     2,
           162, 15638,   435,    23,  1364,   438,  8688,     2,   111, 12121,
          4376,  8666,    31,    63,  6778,  4328,    76,  4376,  8666,   105,
          1468,  1975,  5216,     6]])  tensor([0])
      ```
      Tensor of torch.Size([1, 34])
    

## 2. Multilayer Perceptron Combined With Word Vector

### 2.1 Code

```python
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs
```

## 2.2 Analysis

A Multilayer Perceptron can be concluded as the following diagram.

| Structure |  Description | Statement | Shape |
| :--: | :--: | :--: | :--: |
| Input Layer |  Word embedding | ```self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)``` | (vocab_size, embedding_dim) |
| Word Vector Layer | Map a word (or a token) to  low-dimension, dense and contagious vectors. |```self.linear1 = nn.Linear(embedding_dim, hidden_dim)``` | (embedding_dim, hidden_dim) | 
| (Optional) Combination Layer | Often shows simultaneously with the Word Vector Layer.<br>Here we calculate the average value. | ```embedding = embeddings.mean(dim=1)```| (embedding_dim, hidden_dim) |
| Hidden Layer | Linear transformation:<br>Combination Layer - Hidden Layer | ```hidden = self.activate(self.linear1(embedding))``` | (embedding_dim, hidden_dim) |
| (Optional) Activation Layer | Linear transformation:<br>Hidden Layer - Activation Layer | ``` self.activate = F.relu``` | (embedding_dim, hidden_dim) |
| Output Layer(1) | Linear transformation:<br>Activation Layer - Output Layer| ```self.linear2 = nn.Linear(hidden_dim, num_class)``` | (hidden_dim, num_class) |

In forward:
 Output Layer(2) calculates the log probability of a sequence to every class.  | ```log_probs = F.log_softmax(outputs, dim=1)``` | (hidden_dim, num_class) |

## 3.Training and Testing Based On Multilayer Perceptron

### 3.1 Code

#### 3.1.1 Hyperparameter Configuration
```python
embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5
filter_size = 3
num_filter = 100
```

#### 3.1.2 Data Loading

```python
train_data, test_data, vocab = load_sentence_polarity()
train_dataset = BowDataset(train_data)
test_dataset = BowDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
```

#### 3.1.3 Model Loading

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device)
```

#### 3.1.4 Training

```python
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, offsets, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, offsets)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")
```

#### 3.1.5 Testing

```python
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, offsets, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, offsets)
        acc += (output.argmax(dim=1) == targets).sum().item()

print(f"Acc: {acc / len(test_data_loader):.2f}")
```
```print(f"Acc: {acc / len(test_data_loader):.2f}")```:
Output accuracy on the test dataset. 

### 3.2 Analysis

When testing, the test_dataloader generates a tuple of (inputs,targets) every loop.
An ```input``` is a tensor in the length of corresponding sentence.
A ```target``` is a tensor in the size of torch.Size([1]), which suggests the **predicted class**.


We set batch_size of test dataloader as 1.
The loop goes for 2662 times, which contains 1331 samples with class of "positive",
, and 1331 samples with class of negative. 

Variable ```offsets``` may be relevant to possible mistake, for we do not use **collect function**
as in CNN.

H
The average accuracy is 0.62.

Negative samples as showed as following:

| index | batch | input (to tokens) | target | output_argmax |
| :--: | :--: | :--: | :--: | :--: |
| 0 | 1 |  [allen] manages to breathe life into this somewhat tired premise . | 0 | 1 |
| 1 | 1 | i have two words to say about reign of fire . great dragons ! | 0 | 1 |
| 2 | 1 |  more vaudeville show than well-constructed narrative , but on those terms it's inoffensive and actually rather sweet . | 0 | 1 |
| 385 | 1 | an entertainment so in love with its overinflated mythology that it no longer recognizes the needs of moviegoers for real characters and compelling plots . | 1 | 0 |
| 386 | 1 | borrows from other movies like it in the most ordinary and obvious fashion .  | 1 | 0 |
| 387 | 1 | a chilly , remote , emotionally distant piece . . . so dull that its tagline should be : 'in space , no one can hear you snore . '  | 1 | 0 |

This shows when encoding sentence with word bags, the network only consider the word information, regardless of context information.

## 4. Sentiment Classification Based On CNN
### 4.1 Code

#### 4.1.1 Class
```python
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs
```

#### 4.1.2 Collect Function
```python
def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets
```

#### 4.1.3 Hyper Parameter
```python
filter_size = 3
num_filter = 100
```

### 4.2 Analysis

## 5. Sentiment Classification Based On RNN

### 5.1 Code

#### 5.1.1 Collect Function

```python
def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets
```

#### 5.1.2 Class
```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs
```

### 5.2 Analysis


## 6. Sentiment Classification Based On Transformer

### 6.1 Code

#### 6.1.1 Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
```

#### 6.1.2 Class

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str = "relu"):
        super(Transformer, self).__init__()
        # Word Embedding Layer
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        # Encoding Layer: Using Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Output Layer
        self.output = nn.Linear(hidden_dim, num_class)


    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
```

### 6.2 Analysis







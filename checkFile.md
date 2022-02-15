# 4.6 Motion Classification


Here we introduce motion classification, a practical case study in NLP.

This checkpoint is organized by:
- [0.Data Exploration]()
- [1.Vocabulary Mapping ]()
- [2.Word Vector Layer]()
- [3.Multilayer Perceptron Combined With Word Vector]()
- [4.Data Preprocessing]()
- [5.Training and Testing Based On Multilayer Perceptron]()
- [6.1 Motion Classification Based On CNN]()
- [6.2 Motion Classification Based On RNN]()
- [6.3 Motion Classification Based On Transformer]()

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



## 1. Vocabulary Mapping

### 1.1 Code
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


### 1.2 Experiment




# 2.Word Vector Layer

Map a word (or a token) to  low-dimension, dense and contagious vectors.



## 3.Multilayer Perceptron Combined With Word Vector

### 3.1 Code

```python
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embedding = embeddings.mean(dim=1)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        probs = F.log_softmax(outputs, dim=1)
        return probs

mlp = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
inputs = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
outputs = mlp(inputs)
print(outputs)
```

## 3.2 Analysis

A Multilayer Perceptron can be concluded as the following diagram.

| Structure |  Description | Statement | Shape |
| :--: | :--: | :--: | :--: |
| Input Layer |  Word embedding | ```self.embedding = nn.Embedding(vocab_size, embedding_dim)``` | (vocab_size, embedding_dim) |
| Word Vector Layer | Word vector layer |```self.linear1 = nn.Linear(embedding_dim, hidden_dim)``` | (embedding_dim, hidden_dim) | 
| (Optional) Combination Layer | Often shows simultaneously with the Word Vector Layer.<br>Here we calculate the average value. | ```embedding = embeddings.mean(dim=1)```| (embedding_dim, hidden_dim) |
| Hidden Layer | Linear transformation:<br>Combination Layer - Hidden Layer | ```hidden = self.activate(self.linear1(embedding))``` | (embedding_dim, hidden_dim) |
| (Optional) Activation Layer | Linear transformation:<br>Hidden Layer - Activation Layer | ``` self.activate = F.relu``` | (embedding_dim, hidden_dim) |
| Output Layer(1) | Linear transformation:<br>Activation Layer - Output Layer| ```self.linear2 = nn.Linear(hidden_dim, num_class)``` | (hidden_dim, num_class) |
| Output Layer(2) | Calculate the log probability of a sequence to every class.  | ```probs = F.log_softmax(outputs, dim=1)``` | (hidden_dim, num_class) |


### 3.3 Experiment



## 4. Data Preprocessing

### 4.1 Data Loading

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


### 4.2 Dataset Construction

```python
class BowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
```

### 4.3 Data Transformation

```python
def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets
```

```inputs = pad_sequence(inputs, batch_first=True)```:

Padding samples in a batch to the same length.










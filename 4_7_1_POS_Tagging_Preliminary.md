# 4.7.1 Part-Of-Speech Tagging - Data Preprocessing

This file is the preliminary work in Part-Of-Speech Tagging.

Continuing learning:

- [4.7.2 POS Tagging Based On Feedforward Neural Network]()
- [4.7.3 POS Tagging Based On Recurrent Neural Network - LSTM]()
- [4.7.4 POS Tagging Based On Transformer]()

Part-Of-Speech Tagging can be considered as a task of multi-class text classification.

Input: context of the target word.

Output: class of the target word.

Since 1) the selection of context will not be so large(ie. Besides the target word itself,
we can choose 1-2 words from left or right), 2) positions of context words always play key roles in
tagging the target word, we concatenate word vectors of context as input of multi-layer perceptron.

This method is also called **Window-Based** method.

Visualization of structural data will be showed in the following context.

## 1. Data

### 1.1 Data Exploration

- Original data: 3914 lines of data in the structure of **(sentence,POS tag)**
- Vocab for sentences: 12410
- Classes: 47

| index | sents | postags |
| :--: | :--: | :--: |
| 0 | ('Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.') |('NNP', 'NNP', ',', 'CD', 'NNS', 'JJ', ',', 'MD', 'VB', 'DT', 'NN', 'IN', 'DT', 'JJ', 'NN', 'NNP', 'CD', '.') |
| 1 | ('Mr.', 'Vinken', 'is', 'chairman', 'of', 'Elsevier', 'N.V.', ',', 'the', 'Dutch', 'publishing', 'group', '.') |('NNP', 'NNP', 'VBZ', 'NN', 'IN', 'NNP', 'NNP', ',', 'DT', 'NNP', 'VBG', 'NN', '.') |
| 2 | ('Rudolph', 'Agnew', ',', '55', 'years', 'old', 'and', 'former', 'chairman', 'of', 'Consolidated', 'Gold', 'Fields', 'PLC', ',', 'was', 'named', '*-1', 'a', 'nonexecutive', 'director', 'of', 'this', 'British', 'industrial', 'conglomerate', '.') | ('NNP', 'NNP', ',', 'CD', 'NNS', 'JJ', 'CC', 'JJ', 'NN', 'IN', 'NNP', 'NNP', 'NNP', 'NNP', ',', 'VBD', 'VBN', '-NONE-', 'DT', 'JJ', 'NN', 'IN', 'DT', 'JJ', 'JJ', 'NN', '.') |


### 1.2 Data Preprocessing

#### 1.2.1 Data Loading

```python
def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab
```

A general loading function can be implemented as:
- load corpus
- construct the vocabulary
- map words to integers
- split data into train set and test set

Here we combine **map words to integers** and **split data into train set and test set** in one statement.

#### 1.2.2 Dataset Construction

```python
class LstmDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
```

#### 1.2.3 Data Transformation (Collect Function)

```python
def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])
    return inputs, lengths, targets, inputs != vocab["<pad>"]
```

Here ```ex``` is a sample in the dataset, with the structure of ```(input,targets)```, so we can index the value as:
```python
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]
```


Here a series has more than 1 answer, while in text classification, a token is corresponding to an answer.
```python
    targets = [torch.tensor(ex[1]) for ex in examples]
```

Padding both input series and output series.
```python
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])
```

Addition in return: mask term, which records actually effective tokens of series.
```python
    return inputs, lengths, targets, inputs != vocab["<pad>"]
```


## 2. POS Tagging Based On Feedforward Neural Network

### 2.1 Model

### 2.2 Training And Testing







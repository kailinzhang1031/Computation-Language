# 4.7 Part-Of-Speech Tagging

## 1. Data Processing

### 1.1 Introduction

Part-Of-Speech Tagging can be considered as a task of multi-class text classification.

Input: context of the target word.

Output: class of the target word.

Since 1) the selection of context will not be so large(ie. Besides the target word itself,
we can choose 1-2 words from left or right), 2) positions of context words always play key roles in
tagging the target word, we concatenate word vectors of context as input of multi-layer perceptron.

This method is also called Window-Based method.

Visualization of structural data will be showed in the following context.


### 1.2 Dataset Loading

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

### 1.3 Data Exploration

- Original data: 3914 lines of data in the structure of **(sentence,POS tag)**

| index | sents | postags |
| :--: | :--: | :--: |
| 0 | ('Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.') | 
('NNP', 'NNP', ',', 'CD', 'NNS', 'JJ', ',', 'MD', 'VB', 'DT', 'NN', 'IN', 'DT', 'JJ', 'NN', 'NNP', 'CD', '.') |
| 1 | ('Mr.', 'Vinken', 'is', 'chairman', 'of', 'Elsevier', 'N.V.', ',', 'the', 'Dutch', 'publishing', 'group', '.') |
('NNP', 'NNP', 'VBZ', 'NN', 'IN', 'NNP', 'NNP', ',', 'DT', 'NNP', 'VBG', 'NN', '.') |


## 2. POS Tagging Based On Feedforward Neural Network




## 3. POS Tagging Based On Recurrent Neural Network - LSTM




## 4. POS Tagging Based On Transformer



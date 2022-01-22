# CheckPoint 5.2.3 CBOW

# 0. Introduction
In this section, we will show the implementation of **CBOW** and **Skip-Gram** based
on Pytorch.

Relevant Learning:

How to subclass Dataset in a new class. 

## 1. CBOW

### (1) Data

We create a **CbowDataset** class to subclass **Dataset** module.

**CbowDataset** reserves data from the original data.
All operation is defined in the constructor ```__init__```.

**Instance Attributes Setting:**

Despite for ```self.data```, which reserves the structural data, we can
also create necessary attributes:

```self.bos``` record the beginning of a sentence,

```self.eos``` record the end of a sentence.

**Statements:**

The creation of structural data from original data follows the idea of CBOW.
Here we illustrate operations in an ergodic of sentence, that is to say the
structural data unit is based on a word.

Input: the context of a specific sentence, in the size of context.

Output: current word, also the structural data: **(context, target)**.

**Instantiation**

```python
dataset = CbowDataset(corpus, vocab, context_size=context_size)
```
 We set parameters to load only 1 sentence in corpus, structural data is showed as following: 
```python
[([2, 4, 6, 7], 5), ([4, 5, 7, 8], 6), ([5, 6, 8, 9], 7), ([6, 7, 9, 10], 8), ([7, 8, 10, 11], 9), ([8, 9, 11, 12], 10), ([9, 10, 12, 13], 11), ([10, 11, 13, 14], 12), ([11, 12, 14, 15], 13), ([12, 13, 15, 16], 14), ([13, 14, 16, 17], 15), ([14, 15, 17, 18], 16), ([15, 16, 18, 19], 17), ([16, 17, 19, 9], 18), ([17, 18, 9, 10], 19), ([18, 19, 10, 11], 9), ([19, 9, 11, 10], 10), ([9, 10, 10, 20], 11), ([10, 11, 20, 13], 10), ([11, 10, 13, 21], 20), ([10, 20, 21, 22], 13), ([20, 13, 22, 23], 21), ([13, 21, 23, 24], 22), ([21, 22, 24, 25], 23), ([22, 23, 25, 26], 24), ([23, 24, 26, 27], 25), ([24, 25, 27, 28], 26), ([25, 26, 28, 11], 27), ([26, 27, 11, 29], 28), ([27, 28, 29, 30], 11), ([28, 11, 30, 31], 29), ([11, 29, 31, 19], 30), ([29, 30, 19, 32], 31), ([30, 31, 32, 33], 19), ([31, 19, 33, 34], 32), ([19, 32, 34, 35], 33), ([32, 33, 35, 36], 34), ([33, 34, 36, 37], 35), ([34, 35, 37, 38], 36), ([35, 36, 38, 7], 37), ([36, 37, 7, 39], 38), ([37, 38, 39, 40], 7), ([38, 7, 40, 20], 39), ([7, 39, 20, 41], 40), ([39, 40, 41, 42], 20), ([40, 20, 42, 10], 41), ([20, 41, 10, 3], 42)]
```
The length of this list is 47, which equals to the first sentence in our corpus.

We can see that every word(target) is associated with 4 words, which is twice of the context size.
Meanwhile, word[i] is associated by word[i-2], word[i-1], word[i+1], word[i+2].


#### (2) Model

The model of CbowModel is the same as what we discussed in previous time.

## 2. Skip-gram

### (1) Data

**Instance Attributes Setting**

Instance attributes setting is similar with CBOW.

**Statements**

In **Skip-gram**, we need to create the co-occurence relationship between words,
that is to say **every word** is considered as a **context factor** to predict the
target word, instead of just a sentence.

Here needs further studying.

### (2) Model

The model of Skip-gram is the same as what we discussed in previous time.

## 3. Skip-gram based on Negative Sampling

### (1) Data


When sampling data, for a positive sample, we need to generate a negative sample
according to a specific **probability distribution**.

To ensure the diversity of data, we can call ```collate_fn``` to sample during the loading process.

We will focus on how to **process negative sampling**.

**Instance Attributes Setting**

2 additional variable should be created:

```self.n_negatives```: number of negative samples.

```self.ns_dist```: distribution of negative samples.


**Sampling Function**

4 initial variables are:
```python
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, context_size = contexts.shape
        neg_contexts = []
```

When sampling in a batch, we need to ensure that **the current context** does not contain
**the negative context**.

The structural data unit is (words, contexts, neg_contexts).

### Model

The model of Skip-gram is the same as what we discussed in previous time.

### Training

**Predecessor 1: Mathematics**

Calculate the distribution of the corpus.

Here we use **unigram distribution**.

Calculate the negative sampling distribution according to the unigram distribution.

**Sampling in a batch**

When training in a batch, the process is showed as following:

i. Get the vector representations of word, context, and negative samples.

```python
        word_embeds = model.forward_w(words).unsqueeze(dim=2)
        context_embeds = model.forward_c(contexts)
        neg_context_embeds = model.forward_c(neg_contexts)
```

ii. Log likelihood of positive sample classification.

```python
        context_loss = F.logsigmoid(torch.bmm(context_embeds, word_embeds).squeeze(dim=2))
        context_loss = context_loss.mean(dim=1)
```

iii. Log likelihood of negative samples classification.

```python
        neg_context_loss = F.logsigmoid(torch.bmm(neg_context_embeds, word_embeds).squeeze(dim=2).neg())
        neg_context_loss = neg_context_loss.view(batch_size, -1, n_negatives).sum(dim=2)
        neg_context_loss = neg_context_loss.mean(dim=1)
```

iiii. Log likelihood of loss.
```python
        loss = -(context_loss + neg_context_loss).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
```

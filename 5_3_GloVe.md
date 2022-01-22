# 5.3 Glove

## 0. Introduction

**GloVe** is an unsupervised learning algorithm for obtaining vector representations for words.<sup>**[1]**</sup>

Glove is also based on word-context representation.

## 1. Dataset Subclass ##

### (1) Constructor ###

#### Instance Attributes Setting ####

We first initialize 3 basic attributes:
```python
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
```
```self.cooccur_counts``` records the coocurence time of a pair of (word, context) in given corpus.

#### (2) Statements ####

```python
                w = sentence[i]
                left_contexts = sentence[max(0, i - context_size):i]
                right_contexts = sentence[i+1:min(len(sentence), i + context_size)+1]
                for k, c in enumerate(left_contexts[::-1]):
                    self.cooccur_counts[(w, c)] += 1 / (k + 1)
                for k, c in enumerate(right_contexts):
                    self.cooccur_counts[(w, c)] += 1 / (k + 1)
                self.data = [(w, c, count) for (w, c), count in self.cooccur_counts.items()]
```

For every word and its context, we calculate the counts of cooccurence.

Note that dominator ```k+1``` represents the distance of word and context in (k+1)th co-occurence.

After recording all co-occurences, we can reserve them all in ```self.data```.

### Collate Function ###
We need to return structural data with form of **(words, contexts, counts)**.

## 2. Model ##

### (1) Constructor ###

#### Instance Attributes Setting ####

Initial instance attributes mainly focus on **word embedding and context embedding**.

Despite for original data, we need to consider the **offset vector**.

```python
        # word embedding and offset vector
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        # context embedding and offset vector
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)
```

### (2) Forward Function ###
Here we define 2 forward functions to return embedding vector and offset vector of word and context respectively.

```python
    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)
        return w_embeds, w_biases

    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return c_embeds, c_biases
```

## 3. Training

Training process in every batch follows the idea of **GloVe**.

Step (2)、(3)、(4) are all to calculate the loss.


### (1) Target Extraction

Firstly, we get the embedding & offset vector representation of every (word, context).
```python
        word_embeds, word_biases = model.forward_w(words)
        context_embeds, context_biases = model.forward_c(contexts)
```

### (2) Regression Target Setting
We can also use ```count+1``` to smooth if necessary.
```python
        log_counts = torch.log(counts)
```

### （3） Weight of samples
```python
weight_factor = torch.clamp(torch.pow(counts / m_max, alpha), max=1.0)
```

### (4) Calculate L2-Loss of every sample in a batch
This statement is corresponding to the loss formula of GloVe.

```python
loss = (torch.sum(word_embeds * context_embeds, dim=1) + word_biases + context_biases - log_counts) ** 2
```

### (5) Calculate the weighted loss of every sample
```python
wavg_loss = (weight_factor * loss).mean()
```

# 4. Reference
[1] https://nlp.stanford.edu/projects/glove/




# Check Point 6.2.3 ELMo

# 0. Introduction
This check point is organized by:

**1. Data**

- Data Distribution<br>
- Data Preprocessing```load_corpus```
  - Corpus Loading 
  - Vocabulary Building
  - Corpus Construction
- Dataset Subclass ```class BiLMDataset(Dataset)```
  - Padding
  - Forward and backward inputs and outputs

**2. Model**
  - Highway Embedding ```class Highway(nn.Module)```
    - Constructor
    - Forward Function
  - Convolutional Embedding ```class ConvTokenEmbedder(nn.Module)```
  - ELMo Encoder ```class ELMoLstmEncoder(nn.Module)```
  - BLM Encoding ```class BiLM(nn.Module)```

**3. Training**
- Standard Training Process
- Loss Computation, Accumulation And Clipping Optimization
- PPL computation

**4. Saving**
- Saving BiLM Encoders
- Saving Configurations
- Saving Vocabularies


# 1. Data

## 1.1 Data Distribution

As a case study, we choose 300 sentences in NLTK.reuters as original data.

## 1.2 Data Preprocessing

#### Corpus Loading

We define the function ```load_corpus``` to read raw text file 
and build vocabulary for both words and chars<sup>**[1]**</sup>

We save all sentence in a **list** and save all character in a **dictionary**.

The construction process in a sentence is showed as following:
```python
        for line in tqdm(f):
            tokens = line.rstrip('\n').split(" ")
            # Truncate too-long sentence
            if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
                tokens = line[:max_seq_len-2]
            sent = [BOS_TOKEN]
            for token in tokens:
                # Truncate too-long word.
                if max_tok_len is not None and len(token) + 2 > max_tok_len:
                    token = token[:max_tok_len-2]
                sent.append(token)
                for ch in token:
                    charset.add(ch)
            sent.append(EOS_TOKEN)
            text.append(sent)
```
Here shows the visualization when we have only 1 sentence.
```python
['<bos>', 'asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'u', '.', 's', '.', 'and', 'japan', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 's', 'exporting', 'nations', 'that', 'the', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', 'damage', ',', 'businessmen', 'and', 'officials', 'said', '.\r', '<eos>']
[['<bos>', 'asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'u', '.', 's', '.', 'and', 'japan', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 's', 'exporting', 'nations', 'that', 'the', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', 'damage', ',', 'businessmen', 'and', 'officials', 'said', '.\r', '<eos>']]
```

#### Vocabulary Building

We call ```Vocab.build``` to build word and character vocabulary.

Word-level dictionary ```vocab_w```
```python
['<unk>', '<pad>', '<bos>', '<eos>', '<bos>', 'asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'and', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 'exporting', 'nations', 'that', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', ',', 'businessmen', 'officials', 'said', '.\r', '<eos>']
```

Character-level dictionary```vocab_c```

When building word-level dictionary, we instantiate it as:
```python
 vocab_w = Vocab.build(
        text,
        min_freq=1,
        reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    )
```
When building character-level dictionary, we instantiate it as:

```python
vocab_c = Vocab(tokens=list(charset))
```
The instantiation will only call the constructor.

Word-level vocabulary```vocab_w```

```python
[['<unk>', '<pad>', '<bos>', '<eos>', '<bos>', 'asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'and', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 'exporting', 'nations', 'that', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', ',', 'businessmen', 'officials', 'said', '<eos>']]
```
Character-level vocabulary```vocab_c```
```python
['b', '-', 'w', 'x', 'l', 'u', '<eos>', 's', 'j', 'i', '<pad>', 'n', 'y', 'h', '<bow>', 'd', 'e', 'c', '<bos>', 'p', 'f', 't', 'o', 'r', 'a', 'g', ',', "'", '.', 'm', '<eow>', '<unk>']
```
len(vocab_w) = 45

len(vocab_c) = 32

#### Corpus Construction

The final step is to construct corpus using word_vocab and char_vocab<sup>**[1]**<sup>.

Word-level corpus:
```python
[[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 11, 12, 11, 21, 14, 22, 23, 24, 25, 26, 27, 28, 29, 12, 30, 31, 32, 20, 33, 34, 35, 36, 37, 38, 39, 8, 40, 41, 21, 42, 43, 11, 44]]
```

Character-level corpus:
```python
[[[29, 3, 16], [29, 8, 18, 23, 8, 14, 16], [29, 1, 10, 27, 15, 24, 20, 1, 24, 18, 16], [29, 5, 1, 8, 24, 16], [29, 22, 8, 26, 8, 4, 1, 16], [29, 5, 24, 15, 26, 16], [29, 19, 16], [29, 17, 16], [29, 18, 16], [29, 17, 13, 16], [29, 11, 8, 27, 8, 14, 16], [29, 24, 23, 5, 20, 16], [29, 26, 15, 19, 14, 20, 23, 14, 4, 16], [29, 20, 24, 8, 22, 1, 16], [29, 5, 24, 23, 25, 20, 23, 15, 14, 16], [29, 6, 1, 20, 0, 1, 1, 14, 16], [29, 20, 9, 1, 16], [29, 19, 16], [29, 17, 16], [29, 18, 16], [29, 17, 16], [29, 8, 14, 22, 16], [29, 11, 8, 27, 8, 14, 16], [29, 9, 8, 18, 16], [29, 24, 8, 23, 18, 1, 22, 16], [29, 5, 1, 8, 24, 18, 16], [29, 8, 26, 15, 14, 4, 16], [29, 26, 8, 14, 2, 16], [29, 15, 5, 16], [29, 8, 18, 23, 8, 16], [29, 7, 16], [29, 18, 16], [29, 1, 10, 27, 15, 24, 20, 23, 14, 4, 16], [29, 14, 8, 20, 23, 15, 14, 18, 16], [29, 20, 9, 8, 20, 16], [29, 20, 9, 1, 16], [29, 24, 15, 0, 16], [29, 25, 15, 19, 28, 22, 16], [29, 23, 14, 5, 28, 23, 25, 20, 16], [29, 5, 8, 24, 16], [29, 13, 16], [29, 24, 1, 8, 25, 9, 23, 14, 4, 16], [29, 1, 25, 15, 14, 15, 26, 23, 25, 16], [29, 22, 8, 26, 8, 4, 1, 16], [29, 12, 16], [29, 6, 19, 18, 23, 14, 1, 18, 18, 26, 1, 14, 16], [29, 8, 14, 22, 16], [29, 15, 5, 5, 23, 25, 23, 8, 28, 18, 16], [29, 18, 8, 23, 22, 16], [29, 17, 16], [29, 30, 16]]]
```

len(corpus_w[0]) = len(corpus_c[0]) = 51

**Note that:**

The mapping from word to integers is dynamic, that is to say:

When we check the index of a specific character **a** 
by calling ```vocab_c.convert_tokens_to_ids('a')```,it will be mapped to different integers.


### 1.3 Dataset Subclass

After constructing structural data, we will define Dataset class BiLMDataset.

This class has two targets showed as following:

**Padding:** padding the character and word series to build mini-batch.

**Input and output:** get input and output of BiLM.

Move left series left-side for 1 offset, padding as ```<pad>```.

Move right series right-size for 1 offset, pdding as ```<pad>```.

Prediction will not process in the position of ```<pad>```.

#### A. Constructor

We only set several instance attributes showed in the following:
```python
    def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c):
        super(BiLMDataset, self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data = []
        for sent_w, sent_c in tqdm(zip(corpus_w, corpus_c)):
            self.data.append((sent_w, sent_c))
```

##### B. Collate Function

```python
    def collate_fn(self, examples):
        # lengths: batch_size
        seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])
        # inputs_w
        inputs_w = [torch.tensor(ex[0]) for ex in examples]
        inputs_w = pad_sequence(inputs_w, batch_first=True, padding_value=self.pad_w)
        # inputs_c: batch_size * max_seq_len * max_tok_len
        batch_size, max_seq_len = inputs_w.shape
        print(inputs_w.shape)
        max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])

        inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)
        for i, (sent_w, sent_c) in enumerate(examples):
            for j, tok in enumerate(sent_c):
                inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)

        # fw_input_indexes, bw_input_indexes = [], []
        targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        print(targets_fw[2])
        print(targets_bw[2])
        for i, (sent_w, sent_c) in enumerate(examples):
            targets_fw[i][:len(sent_w)-1] = torch.LongTensor(sent_w[1:])
            targets_bw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w)-1])
        return inputs_w, inputs_c, seq_lens, targets_fw, targets_bw

```

i. Initialize Parameters.

Here we created ```seq_lens``` to **record the length of every sequence**.
```python
        seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])
```

seq_lens = ```tensor([38, 51])```

Note that tensor is put in order, where 51 represents the first tensor,
38 represents the second tensor.

ii. Word Input

When constructing word input, we firstly get all inputs, then padding the word.
```python
        # inputs_w
        inputs_w = [torch.tensor(ex[0]) for ex in examples]
        inputs_w = pad_sequence(inputs_w, batch_first=True, padding_value=self.pad_w)
```

original input:

inputs_ w =
```python
[tensor([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10,
        11, 12, 11, 21, 14, 22, 23, 24, 25, 26, 27, 28, 29, 12, 30, 31, 32, 20,
        33, 34, 35, 36, 37, 38, 39,  8, 40, 41, 21, 42, 43, 44, 45]), 
 tensor([ 4, 46, 47, 48, 49, 50,  5, 51, 52, 10, 11, 12, 11, 53, 54, 14, 55, 56,
        57, 58, 50, 20, 10, 11, 12, 11, 21, 59, 60, 61, 62, 63, 64, 27, 65, 66,
        44, 45])]

```

inputs_w = 
```python
tensor([[ 4, 46, 47, 48, 49, 50,  5, 51, 52, 10, 11, 12, 11, 53, 54, 14, 55, 56,
         57, 58, 50, 20, 10, 11, 12, 11, 21, 59, 60, 61, 62, 63, 64, 27, 65, 66,
         44, 45,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10,
         11, 12, 11, 21, 14, 22, 23, 24, 25, 26, 27, 28, 29, 12, 30, 31, 32, 20,
         33, 34, 35, 36, 37, 38, 39,  8, 40, 41, 21, 42, 43, 44, 45]])

```

iii. Character input

When constructing character input, we firstly need to get necessary
parameters:
```python
        batch_size, max_seq_len = inputs_w.shape
        max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])
```
batch_size = 2
max_seq_len = 51
max_tok_len = 16

Getting the original input and padding.
```python
        inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)
        for i, (sent_w, sent_c) in enumerate(examples):

            for j, tok in enumerate(sent_c):
                inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)
```
Every sentence is convert to a tensor of 51*16.

#### Padding
**(before)**


inputs_w =
```python
[tensor([ 4,  0,  0,  0,  0, 21,  5,  0,  0,  8,  9, 10,  9,  0,  0, 11, 22,  0,
         0,  0, 21, 12,  8,  9, 10,  9, 13,  0,  0,  0,  0,  0,  0, 14, 23,  0,
         9, 20]), 
 tensor([ 4,  0,  0,  6, 19, 16,  0, 12,  0,  0,  0,  0, 21, 12,  0, 17,  0, 18,
        21, 12,  0, 17,  0,  0, 15, 10,  0, 22,  0, 23,  0,  9, 20]), 
 tensor([ 4,  5,  6,  0,  7,  0,  8,  9, 10,  0, 11,  0,  0,  0,  0,  0, 12,  8,
         9, 10,  9, 13, 11,  0,  0,  0,  0,  0, 14,  0, 15, 10,  0,  0, 16, 12,
         0,  0,  0,  0, 17,  0,  0,  7, 18,  0, 13,  0, 19,  9, 20])]

```

**(after)** 

inputs_w = 
```python
tensor([[ 4,  0,  0,  0,  0, 21,  5,  0,  0,  8,  9, 10,  9,  0,  0, 11, 22,  0,
          0,  0, 21, 12,  8,  9, 10,  9, 13,  0,  0,  0,  0,  0,  0, 14, 23,  0,
          9, 20,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 4,  0,  0,  6, 19, 16,  0, 12,  0,  0,  0,  0, 21, 12,  0, 17,  0, 18,
         21, 12,  0, 17,  0,  0, 15, 10,  0, 22,  0, 23,  0,  9, 20,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 4,  5,  6,  0,  7,  0,  8,  9, 10,  0, 11,  0,  0,  0,  0,  0, 12,  8,
          9, 10,  9, 13, 11,  0,  0,  0,  0,  0, 14,  0, 15, 10,  0,  0, 16, 12,
          0,  0,  0,  0, 17,  0,  0,  7, 18,  0, 13,  0, 19,  9, 20]])
```

#### Forward and backward inputs and outputs:

Initialize 2 matrix to convert the input.
```python
        targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
```

targets_fw[1] = 
```python
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1])
```
targets_bw[1] = 
```python
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1])
```

Convert inputs_w to forward and backward:
```python
        for i, (sent_w, sent_c) in enumerate(examples):
            targets_fw[i][:len(sent_w)-1] = torch.LongTensor(sent_w[1:])
            targets_bw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w)-1])

```
targets_fw[1] = 
```python
tensor([ 5,  6,  0,  7,  0,  8,  9, 10,  0, 11,  0,  0,  0,  0,  0, 12,  8,  9,
        10,  9, 13, 11,  0,  0,  0,  0,  0, 14,  0, 15, 10,  0,  0, 16, 12,  0,
         0,  0,  0, 17,  0,  0,  7, 18,  0, 13,  0, 19,  9, 20,  1])
```

targets_bw[1] = 
```python
tensor([ 1,  4,  5,  6,  0,  7,  0,  8,  9, 10,  0, 11,  0,  0,  0,  0,  0, 12,
         8,  9, 10,  9, 13, 11,  0,  0,  0,  0,  0, 14,  0, 15, 10,  0,  0, 16,
        12,  0,  0,  0,  0, 17,  0,  0,  7, 18,  0, 13,  0, 19,  9])
```

targets_fw.shape = torch.Size([3, 51])

targets_bw.shape = torch.Size([3, 51])

targets_fw[1].shape = torch.Size([51])

targets_bw[1].shape = torch.Size([51])

# 2. Model

###  2.1 Highway Embedding

We define **Highway** as a model subclass from ```nn.Module```.

#### Constructor

There are 4 inputs in the constructor, which are: ```self, input_dim, num_layers, activation=F.relu```.

**Instance Attributes Settings**

Only inputs require to be initiated.

```python
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self.activation = activation
```


**Statements**

To our convenience, we put statements and the initiation of layers together.

```python
        self.layers = torch.nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        for layer in self.layers:
            # set bias in the gates to be positive
            # such that the highway layer will be biased towards the input part
            layer.bias[input_dim:].data.fill_(1)

```

#### B. Forward Function

The forward propagation processes according to the main idea of **Highway**.
```python
    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)
            hidden = self.activation(projected_inputs[:, 0:self.input_dim])
            gate = torch.sigmoid(projected_inputs[:, self.input_dim:])
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        return curr_inputs
```

Firstly, we extract the current inputs.

In ergodic of layers, the propagation process is the same:

we compute project value, hidden value, gate, then combine 
with current inputs.

The function will return current inputs finally.

Here we set the parameter as:

`
'num_highways': 2,

'projection_dim': 512,

'hidden_dim': 4096,

then the structure of Highways is showed as following:


```python
(highways): Highway(
      (layers): ModuleList(
        (0): Linear(in_features=2048, out_features=4096, bias=True)
        (1): Linear(in_features=2048, out_features=4096, bias=True)
      )
    )
    (projection): Linear(in_features=2048, out_features=512, bias=True)
  )

```


### 2.1 Convoluntional Embedding
In the convolutional layer, we define **ConvTokenEmbedder** to transform tokens
to embeddings.

This convoluntional layer is based on  character-level.

#### 1. Constructor

##### (1) Input

```self```

```vocab_c```character-level dictionary, 

```char_embedding_dim``` embedding dimension

```char_conv_filters``` convolutional filter(or kernel)

```num_highways``` number of highway layer

```output_dim``` output dimension

##### (2) Instance Attributes Setting
We take weight into consideration in the embedding layer.

```python
        super(ConvTokenEmbedder, self).__init__()
        self.vocab_c = vocab_c

        self.char_embeddings = nn.Embedding(
            len(vocab_c),
            char_embedding_dim,
            padding_idx=vocab_c[pad]
        )
        self.char_embeddings.weight.data.uniform_(-0.25, 0.25)

```

#### (3)  Convolution Process
```self.convolutions = nn.ModuleList()``` Create a convolution list to reserve every ergodic.

Build convolutional network for **every kernel**.
```python
        for kernel_size, out_channels in char_conv_filters:
            conv = torch.nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=True
            )
            self.convolutions.append(conv)
```

Represent the dimension of **concatenated layer** by vectors of several convolutional networks.
```python
        self.num_filters = sum(f[1] for f in char_conv_filters)
        self.num_highways = num_highways
        self.highways = Highway(self.num_filters, self.num_highways, activation=F.relu)
```

Since ELMo vector represents the result of every layer, we need to keep the consistence of all layers of vectors.
```python
self.projection = nn.Linear(self.num_filters, output_dim, bias=True)
```

### 2. Forward Function

The forward propagation process is similar to the process in **2.1.1**.

#### 1. Input
```self```

```inputs``` Input vector

```embeddings=self.char_embeddings(inputs)``` Character-level embedding.

#### 2. Initialization
```python
        batch_size, seq_len, token_len = inputs.shape
        inputs = inputs.view(batch_size * seq_len, -1)
        char_embeds = embeddings
        char_embeds = char_embeds.transpose(1, 2)
```

#### 3. Convolution Process
```python
        conv_hiddens = []
        for i in range(len(self.convolutions)):
            conv_hidden = self.convolutions[i](char_embeds)
            conv_hidden, _ = torch.max(conv_hidden, dim=-1)
            conv_hidden = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)
```

#### 4. Vector Concatenation
```python
        token_embeds = torch.cat(conv_hiddens, dim=-1)
        token_embeds = self.highways(token_embeds)
        token_embeds = self.projection(token_embeds)
        token_embeds = token_embeds.view(batch_size, seq_len, -1)

        return token_embeds
```


### 2.2  ELMo Encoder
In the ELMo Encoder part, we define **ELMoLstmEncoder** to encode embeddings.

#### 1. Constructor

##### (1) Input
```python
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout_prob=0.0
    ):

```

#### (2) Initialization
```python
        super(ELMoLstmEncoder, self).__init__()

        # set projection_dim==input_dim for ELMo usage
        self.projection_dim = input_dim
        self.num_layers = num_layers

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        self.forward_projections = nn.ModuleList()
        self.backward_projections = nn.ModuleList()
```

#### (3) Convolution Process

```python
        lstm_input_dim = input_dim
        for _ in range(num_layers):
            forward_layer = nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            backward_layer = nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            lstm_input_dim = self.projection_dim

```

#### (4) Concatenation
```python
            self.forward_layers.append(forward_layer)
            self.forward_projections.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projections.append(backward_projection)
```
### 2. Forward Function

#### (1) Input
```python
    def forward(self, inputs, lengths):
```

#### (2) Initialization

```python
        batch_size, seq_len, input_dim = inputs.shape
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        for i in range(lengths.shape[0]):
            rev_idx[i,:lengths[i]] = torch.arange(lengths[i]-1, -1, -1)
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1, rev_idx)
```

#### (3) Convolution Process & Concatenation
A forward process and a backward process are paired in one convolution process.
```python
        for layer_index in range(self.num_layers):
            # Transfer `lengths` to CPU to be compatible with latest PyTorch versions.
            packed_forward_inputs = pack_padded_sequence(
                forward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_backward_inputs = pack_padded_sequence(
                backward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # forward
            forward_layer = self.forward_layers[layer_index]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            forward = self.forward_projections[layer_index](forward)
            stacked_forward_states.append(forward)

            # backward
            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
            backward = self.backward_projections[layer_index](backward)
            # convert back to original sequence order using rev_idx
            stacked_backward_states.append(backward.gather(1, rev_idx))

            forward_inputs, backward_inputs = forward, backward
```

```python
class ELMoLstmEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout_prob=0.0
    ):
        super(ELMoLstmEncoder, self).__init__()

        # set projection_dim==input_dim for ELMo usage
        self.projection_dim = input_dim
        self.num_layers = num_layers

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        self.forward_projections = nn.ModuleList()
        self.backward_projections = nn.ModuleList()

        lstm_input_dim = input_dim
        for _ in range(num_layers):
            forward_layer = nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            backward_layer = nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            lstm_input_dim = self.projection_dim

            self.forward_layers.append(forward_layer)
            self.forward_projections.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projections.append(backward_projection)

    def forward(self, inputs, lengths):
        batch_size, seq_len, input_dim = inputs.shape
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        for i in range(lengths.shape[0]):
            rev_idx[i,:lengths[i]] = torch.arange(lengths[i]-1, -1, -1)
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1, rev_idx)

        forward_inputs, backward_inputs = inputs, rev_inputs
        stacked_forward_states, stacked_backward_states = [], []

        for layer_index in range(self.num_layers):
            # Transfer `lengths` to CPU to be compatible with latest PyTorch versions.
            packed_forward_inputs = pack_padded_sequence(
                forward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_backward_inputs = pack_padded_sequence(
                backward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # forward
            forward_layer = self.forward_layers[layer_index]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            forward = self.forward_projections[layer_index](forward)
            stacked_forward_states.append(forward)

            # backward
            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
            backward = self.backward_projections[layer_index](backward)
            # convert back to original sequence order using rev_idx
            stacked_backward_states.append(backward.gather(1, rev_idx))

            forward_inputs, backward_inputs = forward, backward

        # stacked_forward_states: [batch_size, seq_len, projection_dim] * num_layers
        # stacked_backward_states: [batch_size, seq_len, projection_dim] * num_layers
        return stacked_forward_states, stacked_backward_states

```

### 2.3 Bidirectional Encoding

In the BLM part, we define **BiLM** to construct the Bidirectional Language Model.

### (1) Subclass
```python
class BiLM(nn.Module):
```

### (2) Constructor

####  Input
```python
    def __init__(self, configs, vocab_w, vocab_c):
```
####  Initialization

```python
        super(BiLM, self).__init__()
        self.dropout_prob = configs['dropout_prob']
        self.num_classes = len(vocab_w)

        self.token_embedder = ConvTokenEmbedder(
            vocab_c,
            configs['char_embedding_dim'],
            configs['char_conv_filters'],
            configs['num_highways'],
            configs['projection_dim']
        )

        self.encoder = ELMoLstmEncoder(
            configs['projection_dim'],
            configs['hidden_dim'],
            configs['num_layers']
        )

        self.classifier = nn.Linear(configs['projection_dim'], self.num_classes)
```
5 instance attributes requires initialization.

```self.dropout_prob``` DropOut probability

```self.num_classes``` Output classes

```self.token_embedder``` Token embedding layer

```self.encoder``` ELMo embedding layer

```self.classifier``` Linear classifier layer

#### (3) Forward Function
The propagation process is showed as following:

- token embedding
- drop out
- forward and backward encoding
- forward and backward classify, the outputs are the last layers of bidirectional process.

```python
    def forward(self, inputs, lengths):
        token_embeds = self.token_embedder(inputs)
        token_embeds = F.dropout(token_embeds, self.dropout_prob)
        forward, backward = self.encoder(token_embeds, lengths)

        return self.classifier(forward[-1]), self.classifier(backward[-1])

```

#### (4) Saving Method
```python
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pth'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier.pth'))

```

#### (5) Loading Method
```python
    def load_pretrained(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pth')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pth')))
        self.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier.pth')))
```

## 4 Training

### 4.1 Instantiation

```python
configs = {
    'max_tok_len': 50,
    'train_file': './train.txt', # path to your training file, line-by-line and tokenized
    'model_path': './elmo_bilm',
    'char_embedding_dim': 50,
    'char_conv_filters': [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
    'num_highways': 2,
    'projection_dim': 512,
    'hidden_dim': 4096,
    'num_layers': 2,
    'batch_size': 3,
    'dropout_prob': 0.1,
    'learning_rate': 0.0004,
    'clip_grad': 5,
    'num_epoch': 11
}

corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(configs['train_file'])
train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
train_loader = get_loader(train_data, configs['batch_size'])

criterion = nn.CrossEntropyLoss(
    ignore_index=vocab_w[PAD_TOKEN],
    reduction="sum"
)
print("Building BiLM model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLM(configs, vocab_w, vocab_c)
print(model)
model.to(device)

optimizer = optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=configs['learning_rate']
)
```

### 4.2 Training Process

### 4.2.1 Standard Training Process
The training process in a batch follows the standard training process,
except for loss computation, clipping optimization and ppl computation.
```python
model.train()
for epoch in range(configs['num_epoch']):
    total_loss = 0
    total_tags = 0 # number of valid predictions
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        batch = [x.to(device) for x in batch]
        inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = batch

        optimizer.zero_grad()
        outputs_fw, outputs_bw = model(inputs_c, seq_lens)
        loss_fw = criterion(
            outputs_fw.view(-1, outputs_fw.shape[-1]),
            targets_fw.view(-1)
        )
        loss_bw = criterion(
            outputs_bw.view(-1, outputs_bw.shape[-1]),
            targets_bw.view(-1)
        )
        loss = (loss_fw + loss_bw) / 2.0
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['clip_grad'])
        optimizer.step()

        total_loss += loss_fw.item()
        total_tags += seq_lens.sum().item()

    train_ppl = np.exp(total_loss / total_tags)
    print(f"Train PPL: {train_ppl:.2f}")

```

#### 4.2.2 Loss Computation, Accumulation And Clipping Optimization
For convenience to illustrate the process of loss computation, accumulation and clipping optimization, 
we put two process together.
```python
        loss_fw = criterion(
            outputs_fw.view(-1, outputs_fw.shape[-1]),
            targets_fw.view(-1)
        )
        loss_bw = criterion(
            outputs_bw.view(-1, outputs_bw.shape[-1]),
            targets_bw.view(-1)
        )
        loss = (loss_fw + loss_bw) / 2.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['clip_grad'])
        optimizer.step()

        total_loss += loss_fw.item()
        total_tags += seq_lens.sum().item()       
```

#### 4.2.3 PPL computation
```python
    train_ppl = np.exp(total_loss / total_tags)
    print(f"Train PPL: {train_ppl:.2f}")
```

## 5 Saving
We need to save BiLM encoders, configurations and vocabularies.
### 5.1 Saving BiLM Encoders
```python
# save BiLM encoders
model.save_pretrained(configs['model_path'])
```
### 5.2 Saving Configurations
```python
# save configs
json.dump(configs, open(os.path.join(configs['model_path'], 'configs.json'), "w"))
```

### 5.3 Saving Vocabularies
```python
# save vocabularies
save_vocab(vocab_w, os.path.join(configs['model_path'], 'word.dic'))
save_vocab(vocab_c, os.path.join(configs['model_path'], 'char.dic'))
```

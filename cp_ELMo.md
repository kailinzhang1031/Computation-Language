# Check Point 6.2.3 ELMo0

# 1. Data

## (1) Data Distribution

As a case study, we choose 1-3 sentence(s) in NLTK.reuters as original data.

## (2) Dataset Construction

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

#### Corpus Construction

The final step is to construct corpus using word_voab and char_vocab<sup>**[1]**<sup>.

The method return 4 attributes as following:
```python
return corpus_w, corpus_c, vocab_w, vocab_c
```

Visualization

word-level dictionary ```vocab_w```
```python
['<unk>', '<pad>', '<bos>', '<eos>', '<bos>', 'asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'and', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 'exporting', 'nations', 'that', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', ',', 'businessmen', 'officials', 'said', '.\r', '<eos>']
```

Following relationships between variables are satisfied when 1 sentence is loaded from the corpus.

len(vocab_w) = len(corpus_w[0]) - 5,
where 5 represents ```'<unk>', '<pad>', '<bos>', '<eos>', '<bos>',```

len(corpus_w) = 1,

len(corpus_c[0]) = 51

len(corpus_w[0]) = 51

len(vocab_w) = 46

len(vocab_c) = 33


**A. Original Data**
```python
asian exporters fear damage from u . s .- japan rift mounting trade friction between the u . s . and japan has raised fears among many of asia ' s exporting nations that the row could inflict far - reaching economic damage , businessmen and officials said .
```

## (3) Dataset Subclass
After constructing structural data, we will define Dataset class
BiLMDataset.

This class has two targets showed as following:

**Padding:** padding the character and word series to build mini-batch.

**Input and output:** get input and output of BiLM.

Move left series left-side for 1 offset, padding as ```<pad>```.

Move right series right-size for 1 offset, pdding as ```<pad>```.

Prediction will not process in the position of ```<pad>```.

#### A. Constructor

We only set several instance attributes showed in the following:
```python
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data = []
        for sent_w, sent_c in tqdm(zip(corpus_w, corpus_c)):
            self.data.append((sent_w, sent_c))
```

##### B. Collate Function

i. Initialize Parameters.

Here we created ```seq_lens``` as offset vector in a batch.
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
        33, 34, 35, 36, 37, 38, 39,  8, 40, 41, 21, 42, 43, 44, 45]), tensor([ 4, 46, 47, 48, 49, 50,  5, 51, 52, 10, 11, 12, 11, 53, 54, 14, 55, 56,
        57, 58, 50, 20, 10, 11, 12, 11, 21, 59, 60, 61, 62, 63, 64, 27, 65, 66,
        44, 45])]

```

padding:

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

Padding:
(before)
inputs_w =
```python
[tensor([ 4,  0,  0,  0,  0, 21,  5,  0,  0,  8,  9, 10,  9,  0,  0, 11, 22,  0,
         0,  0, 21, 12,  8,  9, 10,  9, 13,  0,  0,  0,  0,  0,  0, 14, 23,  0,
         9, 20]), tensor([ 4,  0,  0,  6, 19, 16,  0, 12,  0,  0,  0,  0, 21, 12,  0, 17,  0, 18,
        21, 12,  0, 17,  0,  0, 15, 10,  0, 22,  0, 23,  0,  9, 20]), tensor([ 4,  5,  6,  0,  7,  0,  8,  9, 10,  0, 11,  0,  0,  0,  0,  0, 12,  8,
         9, 10,  9, 13, 11,  0,  0,  0,  0,  0, 14,  0, 15, 10,  0,  0, 16, 12,
         0,  0,  0,  0, 17,  0,  0,  7, 18,  0, 13,  0, 19,  9, 20])]

```

(after) inputs_w = 
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



Forward and backward inputs:

Initialize 2 matrix to convert the input.
```python
        targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
```
targets_fw.shape = torch.Size([3, 51])
targets_bw.shape = torch.Size([3, 51])
targets_fw[1].shape = torch.Size([51])
targets_bw[1].shape = torch.Size([51])

targets_fw[1] = 
```python
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1])
```
targets_bw[2] = 
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

targets_fw[1].shape = torch.Size([51])
targets_bw[1].shape = torch.Size([51])


# CheckPoint 5.1.3

## 0. Introduction

## 1. Data Preparation

### (1) Data Distribution

NLTK Requters corpus

data length  = 54716

### (2) Load oringinal data
standard format:

```python
['ASIAN', 'EXPORTERS', 'FEAR', 'DAMAGE', 'FROM', 'U', '.', 'S', '.-', 'JAPAN', 'RIFT', 'Mounting', 'trade', 'friction', 'between', 'the', 'U', '.', 'S', '.', 'And', 'Japan', 'has', 'raised', 'fears', 'among', 'many', 'of', 'Asia', "'", 's', 'exporting', 'nations', 'that', 'the', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', 'damage', ',', 'businessmen', 'and', 'officials', 'said', '.']
```
ASIAN EXPORTERS FEAR DAMAGE FROM U . S .- JAPAN RIFT Mounting trade friction between the U . S . And Japan has raised fears among many of Asia ' s exporting nations that the row could inflict far - reaching economic damage , businessmen and officials said .

length = 49

### (3) Build a vocabulary

Vocab reserves every word (or token) in the whole text without repeating.
every token is mapped to an interger,

### (4) Convert every token to id

```python
['asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'u', '.', 's', '.', 'and', 'japan', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 's', 'exporting', 'nations', 'that', 'the', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', 'damage', ',', 'businessmen', 'and', 'officials', 'said', '.']
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 9, 10, 11, 10, 20, 13, 21, 22, 23, 24, 25, 26, 27, 28, 11, 29, 30, 31, 19, 32, 33, 34, 35, 36, 37, 38, 7, 39, 40, 20, 41, 42, 10]
```
Here we can call method of ```vocab```:
```python
vocab.__getitem__('asian')
```

## 2. Feed Forward Neural Network

### (1) Dataset

### 1. Data Structure
when ergodic all sentence, every sentence is represented as follows:
Original Tokens:
```python
['asian', 'exporters', 'fear', 'damage', 'from', 'u', '.', 's', '.-', 'japan', 'rift', 'mounting', 'trade', 'friction', 'between', 'the', 'u', '.', 's', '.', 'and', 'japan', 'has', 'raised', 'fears', 'among', 'many', 'of', 'asia', "'", 's', 'exporting', 'nations', 'that', 'the', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', 'damage', ',', 'businessmen', 'and', 'officials', 'said', '.']
['they', 'told', 'reuter', 'correspondents', 'in', 'asian', 'capitals', 'a', 'u', '.', 's', '.', 'move', 'against', 'japan', 'might', 'boost', 'protectionist', 'sentiment', 'in', 'the', 'u', '.', 's', '.', 'and', 'lead', 'to', 'curbs', 'on', 'american', 'imports', 'of', 'their', 'products', '.']
```
Sentence ergodic:

10 sentences are showed as follows:
```python
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 9, 10, 11, 10, 20, 13, 21, 22, 23, 24, 25, 26, 27, 28, 11, 29, 30, 31, 19, 32, 33, 34, 35, 36, 37, 38, 7, 39, 40, 20, 41, 42, 10]
[43, 44, 45, 46, 47, 4, 48, 49, 9, 10, 11, 10, 50, 51, 13, 52, 53, 54, 55, 47, 19, 9, 10, 11, 10, 20, 56, 57, 58, 59, 60, 61, 26, 62, 63, 10]
[64, 65, 5, 42, 31, 66, 19, 67, 68, 69, 70, 47, 19, 71, 36, 72, 39, 47, 19, 73, 36, 74, 75, 28, 11, 76, 52, 77, 62, 78, 10]
[19, 9, 10, 11, 10, 21, 42, 79, 80, 81, 82, 83, 84, 26, 85, 59, 61, 26, 86, 87, 88, 59, 89, 90, 39, 47, 91, 92, 13, 28, 11, 93, 94, 57, 95, 57, 49, 96, 97, 57, 98, 99, 59, 100, 101, 102, 103, 104, 10]
[105, 86, 106, 107, 19, 108, 26, 19, 85, 102, 109, 110, 84, 20, 111, 92, 112, 87, 113, 42, 43, 68, 114, 115, 116, 26, 63, 117, 118, 19, 119, 120, 10]
[121, 122, 123, 28, 124, 77, 125, 57, 126, 127, 128, 42, 49, 129, 92, 130, 86, 87, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 10, 124, 141]
[121, 142, 19, 85, 143, 47, 144, 92, 145, 146, 26, 147, 148, 49, 149, 150, 79, 80, 151, 19, 152, 153, 26, 116, 154, 26, 88, 155, 57, 85, 156, 57, 19, 9, 10, 11, 157, 42, 158, 159, 39, 49, 160, 161, 102, 19, 75, 162, 26, 163, 137, 138, 139, 164, 165, 20, 135, 141]
[47, 166, 39, 40, 20, 41, 167, 168, 169, 10]
[121, 122, 167, 170, 26, 19, 171, 26, 19, 9, 10, 11, 10]
[172, 51, 13, 173, 79, 174, 175, 49, 176, 57, 177, 128, 42, 49, 178, 179, 16, 180, 181, 182, 97, 57, 77, 183, 10]
```

Note that:

Here ```self.data``` saved context information of every word, that is to say:

```len(self.data) = number of word```

### 2. DataLoader

Dataloader can be defined int a  function, in the method, we can instantiate DataLoader,

parameter includes ```(dataset, batch_size, shuffle=True)```

In the function, we can subclass Dataloader: 

parameter of class includes:
    ``` dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle```

Here we also need to define the ```collate_fn```, which converts original data (with structure) to samples.

Note that: Instantiation of DataLoader **do not** implicitly call ```collate_fn```.

### (3) Training

#### 1. Input

batch_size defines number of samples that will be propagated through the network.</sup>**[1]**</sup>

i.e. when batch_size = 1, in an epoch,

number of training (or propagation) = size of data / batch_size.

In each training process, input sample is in the form of```(input,traget)```, like:

```(tensor([[89, 90]]), tensor([39]))```

if we put all data in just one training process, then the length of input and output = size of all data (batch_size = 1)


#### 2. Common Training Process

A common training process is showed as follows:

Here we define a process (or a propagation) in the context of a batch in an epoch.

```python
inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
```
We can inllustrate the process as:

**(1) Sample Creation:** In the form of (input, target).

**(2) Zero Gradient:** Sets the gradients of all optimized torch.Tensor s to zero. (By **Pytorch Doc**)

**(3) Probability Computation:** Always process as log.

**(4) Loss Computation:** Loss between probability and the true target.

**(5) Backward Propagation:** Computes the gradient of current tensor w.r.t. graph leaves. (By **Pytorch Doc**)

**(6) Optimization:** Performs a single optimization step (parameter update).(By **Pytorch Doc**)

**(7) Loss Accumulation:** Loss in an epoch should be accumutated in every propagation, which is a float value and can be got by
```loss.item()```, Here ```loss``` is in the type of ```<class 'torch.Tensor'>```.

#### 3. Embedding saving

We can create a function like ```save_pretrained(vocab, embeds, save_path)```.

Here "Save pretrained token vectors in a unified format, where the first line 
specifies the `number_of_tokens` and `embedding_dim` followed with all
token vectors, one token per line." (From source code)

**[1]** itdxer (https://stats.stackexchange.com/users/64943/itdxer), What is batch size in neural network?, URL (version: 2019-04-05): https://stats.stackexchange.com/q/153535



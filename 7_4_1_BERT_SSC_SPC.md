# CheckPoint 7.4 Applications of BERT_1

# 0. Introduction

Here we will introduce several applications on BERT. 

Regarding that BERT requires highcomputability, which is hard for laptops, we will only introduce processes like data preprocessing,
dataset construction, code interpretation, and other necessary work, except for detailed
training process and analysis.

In this checkpoint, we will introduce 2 classification tasks, which are **Single Sentence Classification(SSC)** 
and **Sentence Pair Classification(SPC)**.

In the next checkpoint, we will introduce generation task **Span-extracting Reading Comprehension(SRC)**
and series tokenization task **Named Entity Recognition(NER)**.

# 1. Single Sentence Classification

## 1.1 Data

### 1.1.1 Data Distribution

The Stanford Sentiment Treebank is the first corpus with fully labeled parse trees
that allows for a complete analysis of the compositional effects of sentiment in 
language.<sup>**[1]**</sup>

This dataset contains 3 subsets, which is train set: 67349 items, 
validation set: 872 items and test set: 1821 items respectively.

An instance of an item is showed as following:

```python
{   
    sentence:'The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .'
    label:0.694440007209777
    tokens: 'The|Rock|is|destined|to|be|the|21st|Century|'s|new|``|Conan|''|and|that|he|'s|going|to|make|a|splash|even|greater|than|Arnold|Schwarzenegger|,|Jean-Claud|Van|Damme|or|Steven|Segal|.'
    tree: '70|70|68|67|63|62|61|60|58|58|57|56|56|64|65|55|54|53|52|51|49|47|47|46|46|45|40|40|41|39|38|38|43|37|37|69|44|39|42|41|42|43|44|45|50|48|48|49|50|51|52|53|54|55|66|57|59|59|60|61|62|63|64|65|66|67|68|69|71|71|0'

}
```

For more details, please click here.

### 1.1.2 Data Preprocessing

Here we use models from HuggingFace.

#### Loading

```python
import numpy as np
import torch.cuda
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

dataset = load_dataset('glue', 'sst2',cache_dir='D:\Program Data\Dataset')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
metric = load_metric('glue', 'sst2')
```
First of all, we need to load the dataset, tokenizer, model and metric.

We can also partially loading dataset. For detailed analysis, please click here.

#### Tokenization


We can tokenize our data by instance of Tokenizer, which only requires parameters
like ```truncation``` and ```padding```.

We just tokenize sentence of the dataset, which is a column in excel form,
or a value of dictionary's key sentence.

Visualizations on some instances are showed as following:


before:
```python
{   
    sentence:'The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .'
    label:0.694440007209777
    tokens: 'The|Rock|is|destined|to|be|the|21st|Century|'s|new|``|Conan|''|and|that|he|'s|going|to|make|a|splash|even|greater|than|Arnold|Schwarzenegger|,|Jean-Claud|Van|Damme|or|Steven|Segal|.'
    tree: '70|70|68|67|63|62|61|60|58|58|57|56|56|64|65|55|54|53|52|51|49|47|47|46|46|45|40|40|41|39|38|38|43|37|37|69|44|39|42|41|42|43|44|45|50|48|48|49|50|51|52|53|54|55|66|57|59|59|60|61|62|63|64|65|66|67|68|69|71|71|0'

}
```
after:
```python
{
    sentence:'The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .'
    label:0.694440007209777
    tokens: 'The|Rock|is|destined|to|be|the|21st|Century|'s|new|``|Conan|''|and|that|he|'s|going|to|make|a|splash|even|greater|than|Arnold|Schwarzenegger|,|Jean-Claud|Van|Damme|or|Steven|Segal|.'
    tree: '70|70|68|67|63|62|61|60|58|58|57|56|56|64|65|55|54|53|52|51|49|47|47|46|46|45|40|40|41|39|38|38|43|37|37|69|44|39|42|41|42|43|44|45|50|48|48|49|50|51|52|53|54|55|66|57|59|59|60|61|62|63|64|65|66|67|68|69|71|71|0'
    
```
```python
input_ids:[  101  1109  2977  1110 17348  1106  1129  1103  6880  5944   112   188
  1207   169   169 17727   112   112  1105  1115  1119   112   188  1280
  1106  1294   170 24194  1256  3407  1190  7296 20452 24156 11819  7582
  9146   117  2893   118   140 15554  1181  3605  8732  3263  1137  6536
 17979  1233   119   102     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0]
```
```python
token_type_ids:[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```
```python


attention_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
}
```
- Tree in every row is created based on parse tree.
- input_ids,token_type_ids, Attention mask is a matrix with size of 1*512.
- input_ids represents sentence as a series of indexes, which are corresponding
to **positions in the vocabualry of input_tokens** .
- attention mask is a matrix filled with 0-1.
- For more details on attention mask, please click here.

#### 1.1.3 Dataset Construction

```python
# 将数据集格式化为torch.Tensor类型以训练PyTorch模型
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)
```
Here we reformat dataset with the format of ```['input_ids', 'token_type_ids', 'attention_mask', 'labels']```,
which is the target data structure.


### 1.2 Traning Process

#### 1.2.1 Metric Definition
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)
```

We set all progress as function, which is even clearer and suitable for training process.

#### 1.2.2 Parameter Initialization

```python
# Default optimizer: Adam
args = TrainingArguments(
    "ft-sst2",                          # output path, saving check points and other output files
    evaluation_strategy="epoch",        # evaluate after every epoch
    learning_rate=2e-5,                 # Initial learning rate
    per_device_train_batch_size = 1,    # batch size for training
    per_device_eval_batch_size=1,      # batch size for validation
    num_train_epochs=1,                 # number of epochs
)
```
Here we instantiate an object from ```TrainingArguments```. 

For detailed illustration, please click here.

#### 1.2.3 Model Instantiation
```python
# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model, # model specification
    args, # parameters
    train_dataset=encoded_dataset["train"], # training set
    eval_dataset=encoded_dataset["validation"], # validation set
    tokenizer=tokenizer, # tokenizer
    compute_metrics=compute_metrics # compute metrics
)


```
Here we instantiate a model obeject from ```Trainer```.
Parameters of the instantiation including:
```model```,
```args```,
```train_dataset```,
```eval_dataset```,
```tokenizer```,
```compute_metrics```.

For detailed illustration, please click here.

#### 1.2.4 Training and Evaluation
```python
trainer.train()
trainer.evaluate()

otuput = {
    'epoch': 2
    'eval_accuracy': 0.7350917431192661
    'eval_loss': 0.9351930022239685
}

```

After instantiation, we can train the model by calling ```trainer.train()```.
After training, we can validate the model on the validation set by calling ```trainer.evaluate()```.

For detailed illustration, please click here.

## 2. Sentence Pair Classification

### 2.1 Data

#### 2.1.1 Data Distribution

The Recognizing Textual Entailment (RTE) datasets come from a series of 
annual textual entailment challenges. The authors of the benchmark combined 
the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim et al., 2006), 
RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009). 
Examples are constructed based on news and Wikipedia text. 

The authors of the benchmark convert all datasets to a two-class split, 
where for three-class datasets they collapse neutral and contradiction into 
not entailment, for consistency<sup>[1]</sup>.

Sample data instances are showed as following:
```python
{
    sentence1: No Weapons of Mass Destruction Found in Iraq Yet.
    sentence2: Weapons of Mass Destruction Found in Iraq.
    label: 1
    idx: 0
}

```

```python
{
    sentence1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.
    sentence2: Pope Benedict XVI is the new leader of the Roman Catholic Church.
    label: 0
    idx: 1
}
```
This dataset also consists of 3 subsets, which are:

train: 2590

validation: 277

test: 30000

For detailed illustration, please click here.


#### 2.1.2 Data Preprocessing

The source code is simillar to what we discussed in SSC, which we will skip
illustration here.

##### Loading

```python

dataset = load_dataset('glue', 'sst2',cache_dir='D:\Program Data\Dataset')


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
metric = load_metric('glue', 'sst2')


```

#### Tokenization

```python

def tokenize(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')
dataset = dataset.map(tokenize, batched=True)
encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

```

##### Dataset Constrution

```python
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)
```


#### 2.2 Training Process

#### 2.2.1 Metric Definition

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)
```


#### 2.2.2 Parameter Initialization

```python
args = TrainingArguments(
    "ft-sst2",                          # 
    evaluation_strategy="epoch",        # 
    learning_rate=2e-5,                 # 
    per_device_train_batch_size = 16,     # 
    per_device_eval_batch_size=16,      # 
    num_train_epochs=1,                 # 
)

```


#### 2.2.3 Model Instantiation

```python
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

```

#### 2.2.4 Training and Evaluation

```python
trainer.train()
trainer.evaluate()

{
    'epoch': 2
    'eval_accuracy': 0.5270758122743683
    'eval_loss': 0.693526139259338
}
```












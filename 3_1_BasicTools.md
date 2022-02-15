# 3.1 Basic Toolkits

Here we introduce toolkits based on NLTK, click here to get corpus and dataset in NLTK: [NLTK Data]().

## 1. Sentence Segment

Segment a text into a list of sentence level.

```python
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

text = gutenberg.raw("austen-emma.txt")
sentences = sent_tokenize(text)
sentence = sentences[100]
```

```python
Mr. Knightley loves to find fault with me, you know--\nin a joke--it is all a joke.
```

Variables:
- ```text```: click here for details of raw data: [gutenberg text]().
- ```sentences```: list of 7493.
- ```sentence```: list of 82.

## 2. Tokenization

Tokenize sentence to a list of word level.

```python
from  nltk.tokenize import  word_tokenize
words = word_tokenize(sentences[100])
```

```python
['Mr.', 'Knightley', 'loves', 'to', 'find', 'fault', 'with', 'me', ',', 'you', 'know', '--', 'in', 'a', 'joke', '--', 'it', 'is', 'all', 'a', 'joke', '.']
```

## 3. Part-Of-Speech Tagging

Tag the part-of-speech of every word.

Tokenization and tagging.
```python
from  nltk import pos_tag
pos = pos_tag(word_tokenize('They sat by the fire.'))
```

```python
[('They', 'PRP'), ('sat', 'VBP'), ('by', 'IN'), ('the', 'DT'), ('fire', 'NN'), ('.', '.')]
```




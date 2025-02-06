# NanoGPT  
*A minimal re-implementation of OpenAI's GPT pre-training algorithms*  

## Overview  
The model is trained on a **300,000-token (1,075,394 characters)** excerpt containing the complete works of Shakespeare (`input.txt`).  
It has not been fine-tuned yet; it will complete your inputs with Shakespeare-sounding nonsense.

## Implementation  

### `gpt-train.py`  
Implements a simple **10M parameter** decoder-only transformer (Vaswani et al., 2017) for **character-level language modeling**, with a context size of `n`.  

### `bigrams.py`  
Trains a **bigram-based language model**, where each token predicts the next token in a sequence (`n=1`).  

# nanogpt
A minimal re-implementation of OpenAI's GPT pre-training algorithms

The model is trained on a 300,000-token (1,075,394 characters) excerpt containing the complete works of Shakespeare `input.txt`. The model has not been fine-tuned yet; it will complete your inputs with Shakespeare-sounding nonsense.

`gpt-train.py` implements a simple 10M parameter decoder-only transformer (Vaswani et al., 2017) for character-level language modeling, with a size `n` context size.
`bigrams.py` is trained using a bigram-based approach, where each token predicts the next token in a sequence (`n=1`)

# Example of usage

## Procrustes

python3 train_procrustes.py corpora/unita_corpus unita

unita : corpus name

corpora/unita_corpus : corpus path

python3 train_procrustes.py corpora/lastampa_corpus lastampa

lastampa : corpus name

corpora/lastampa_corpus : corpus path

It computes the word vectors for diachronic corpora and align them with Orthogonal Procrustes.


## Spearman Correlation


python3 spearman_corr.py unita unita_vectors lastampa lastampa_vectors

unita : corpus name

lastampa : corpus name

unita_vectors : folder where vectors are stored for the first corpus

lastampa_vectors : folder where vectors are stored for the second corpus

It computes the spearman correlation and the lexical semantic changes (stored in the sims folder).

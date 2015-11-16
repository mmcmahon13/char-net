# char-net
Molly and David's Term NLP project - the goal of this model is to automatically infer character types, as well as network of character alliances, from raw novel text. Our approach is heavily reliant on the previous work of Bamman, et. al. and others, but we propose a simpler approach for character persona inference using only topic modeling and clustering, as well as a new set of character features.

Our code loads the JSON output of Bamman et. al.'s BookNLP pipeline, then performs LDA topic modeling using the derived character features. For each character, a vector of possible topics and their probabilities is produced; we use these vectors to perform KMeans clustering, producing groups of characters with similar actions.

We specifically demonstrate our approach using J.K. Rowling's Harry Potter series - our code is not yet generalized to work on other inputs, but we hope to fix this in the future.

### Citations (for tools used):

##### Book-NLP:

David Bamman, Ted Underwood and Noah Smith, "A Bayesian Mixed Effects Model of Literary Character," ACL 2014.




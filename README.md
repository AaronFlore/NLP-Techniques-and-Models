# NLP-Techniques-and-Models
Contains a variety of Natural Language Processing (NLP) programs and models.

## Sentiment-Classifier-and-Text-Generator
### lm.py
Implements a bigram language model with add-alpha smoothing to generate text and evaluate its performance using perplexity. It tokenizes input text, constructs a vocabulary, computes bigram probabilities, and can generate text sequences based on learned patterns. The script also includes functionality to plot word frequency distributions and perplexity metrics. It is configurable via command-line arguments, allowing users to specify training and validation data, the smoothing parameter, and plotting options.
#### sentiment_classifier.py
Implements several classifiers to predict sentiment. It includes a baseline classifier that always predicts positive sentiment, a Naive Bayes classifier, a Logistic Regression classifier, and a Bigram Logistic Regression classifier. The script utilizes nltk for text preprocessing and scikit-learn for model training and evaluation. It supports command-line arguments to specify training and validation data, remove stopwords, and configure logistic regression parameters. The classifiers are evaluated on their accuracy in predicting sentiment labels for given sentences.

## Text-Classification
### classifier.py
Implements hate speech detection based on the work by Davidson et al., utilizing natural language processing and machine learning techniques. It preprocesses tweet data, extracts various features including sentiment scores and Twitter-specific features, and trains a logistic regression classifier. The classifier's performance is evaluated, and classification report metrics are printed to assess its effectiveness in predicting hate speech.

### nn.py
Implements various neural network models for text classification, including feedforward, LSTM, GRU, and bidirectional LSTM. It preprocesses text data, loads word embeddings, and trains the selected model using PyTorch. The models are evaluated on a test dataset, and classification reports are printed to assess their performance in classifying text data.

## Named-Entity-Recognition
### hmm.py
Implements a Hidden Markov Model (HMM) for named entity recognition using greedy and Viterbi decoding algorithms. It loads training and test data, trains the HMM model, and evaluates its performance on the test data. The script provides options to specify the decoding type (either 'viterbi' or 'greedy') and whether to lowercase the text. The evaluation results include classification reports indicating accuracy and other metrics for each named entity tag.

### nn.py
implements a neural network model for named entity recognition (NER) using PyTorch. It includes options to specify various hyperparameters such as learning rate, embedding dimension, hidden layer dimension, and number of training epochs. The script loads training and test data, builds and trains the neural network model, and evaluates its performance using classification reports. The model architecture consists of an embedding layer, followed by an LSTM layer, and finally a linear layer for classification. The training process is conducted using the Adam optimizer and the CrossEntropyLoss function.

# Installation
1. Navigate to one of the 3 folders
2. ```bash
   pip install -r requirements.txt
   ```
3. Run the python scripts
4. Please note: nn.py requires an additional file, glove.6B.50d.txt, which can be downloaded from [stanford's nlp website.](https://nlp.stanford.edu/projects/glove/)

import string
from pos_hmm import HMM

def preprocess_sentence(sentence, vocab):
    # normalization
    tokens = [word.lower().strip(string.punctuation) for word in sentence.split()]
    return [token if token in vocab else '<UNK>' for token in tokens]

# dataset 
tagged_sentences = [
    [(word.lower(), tag) for word, tag in sent] for sent in [
        [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
        [("A", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
        [("The", "DET"), ("dog", "NOUN"), ("sleeps", "VERB")],
        [("My", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
        [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB"), ("loudly", "ADV")],
        [("Your", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
        [("The", "DET"), ("bird", "NOUN"), ("sings", "VERB"), ("sweetly", "ADV")],
        [("A", "DET"), ("bird", "NOUN"), ("chirps", "VERB")]
    ]
]

# HMM model and training
model = HMM()
model.train(tagged_sentences)

# Build a vocabulary including an unknown token
vocab = set()
for sent in tagged_sentences:
    for word, _ in sent:
        vocab.add(word)
vocab.add('<UNK>')

# Interactive testing loop
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    test_sentence = preprocess_sentence(user_input, vocab)
    predicted_tags = model.viterbi(test_sentence)

    print("Preprocessed Sentence:", test_sentence)
    print("Predicted Tags:", predicted_tags)
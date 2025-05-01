from pos_hmm import HMM

# Training data
tagged_sentences = [
    [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("A", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
    [("The", "DET"), ("dog", "NOUN"), ("sleeps", "VERB")],
    [("My", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB"), ("loudly", "ADV")],
    [("Your", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("The", "DET"), ("bird", "NOUN"), ("sings", "VERB"), ("sweetly", "ADV")],
    [("A", "DET"), ("bird", "NOUN"), ("chirps", "VERB")]
]

# HMM model and training
model = HMM()
model.train(tagged_sentences)

# Test sentences
test_sentences = [
    "The cat meows",
    "My dog barks loudly"
]

# Viterbi algorithm for each test sentence
for sentence in test_sentences:
    words = sentence.split()
    predicted_tags = model.viterbi(words)
    print("Sentence:", sentence)
    print("Predicted Tags:", predicted_tags)
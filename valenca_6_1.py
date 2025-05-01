from pos_hmm import HMM

# dataset
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

# Interactive testing loop
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    test_sentence = user_input.split()
    predicted_tags = model.viterbi(test_sentence)

    print("Sentence:", test_sentence)
    print("Predicted Tags:", predicted_tags)
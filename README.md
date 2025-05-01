# Unit 6: Hidden Markov Models for Part-of-Speech Tagging  
**Subject:** Natural Language Processing (CCS 249)  
**West Visayas State University**

## Overview

This unit demonstrates the implementation and application of a Hidden Markov Model (HMM) for Part-of-Speech (PoS) tagging using Python. The exercises guide you through:

- Structuring a tagged dataset for training
- Training an HMM on the dataset
- Using the Viterbi algorithm to predict the best PoS sequence for new sentences
- Interactive and batch testing of the trained model

## Files

- **pos_hmm.py**  
  Contains the `HMM` class implementation, including training and Viterbi decoding.

- **valenca_6_1.py**  
  Interactive script:  
  - Trains the HMM on a small tagged dataset  
  - Allows the user to input sentences and outputs predicted PoS tags

- **valencia_6_2.py**  
  Batch script:  
  - Trains the HMM on the same dataset  
  - Runs the Viterbi algorithm on predefined test sentences and prints the results

- **main.py**  
  Example script showing basic usage of the HMM class with a small dataset.

## Dataset

The training dataset consists of simple English sentences, each word tagged with its PoS:

```
The_DET cat_NOUN sleeps_VERB
A_DET dog_NOUN barks_VERB
The_DET dog_NOUN sleeps_VERB
My_DET dog_NOUN runs_VERB fast_ADV
A_DET cat_NOUN meows_VERB loudly_ADV
Your_DET cat_NOUN runs_VERB
The_DET bird_NOUN sings_VERB sweetly_ADV
A_DET bird_NOUN chirps_VERB
```

## How to Use

1. **Training the Model:**  
   The scripts automatically train the HMM using the provided tagged sentences.

2. **Testing:**
   - Run `valenca_6_1.py` for interactive testing.  
     Enter a sentence (e.g., `The cat meows`) and see the predicted tags.
   - Run `valencia_6_2.py` to see predictions for the test sentences:
     - "The cat meows"
     - "My dog barks loudly"

3. **Custom Testing:**  
   You can modify the test sentences or tagged data as needed.

## Requirements

- Python 3.x
- No external libraries required (uses only standard Python and `numpy`)

---
*Prepared for CCS 249 - Natural Language Processing, Unit 6*

from collections import defaultdict

class HMM:
    def __init__(self):
        self.states = set()
        self.vocab = set()
        self.start_probs = defaultdict(float)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        self.emit_probs = defaultdict(lambda: defaultdict(float))

    def train(self, tagged_sentences):
        start_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        tag_counts = defaultdict(int)

        for sentence in tagged_sentences:
            for i, (word, tag) in enumerate(sentence):
                self.states.add(tag)
                self.vocab.add(word)
                tag_counts[tag] += 1
                emission_counts[tag][word] += 1

                if i == 0:
                    start_counts[tag] += 1
                else:
                    prev_tag = sentence[i-1][1]
                    transition_counts[prev_tag][tag] += 1

        total_starts = sum(start_counts.values())
        # Compute probabilities with optional smoothing
        smoothing = 1e-6
        for tag in self.states:
            self.start_probs[tag] = (start_counts[tag] + smoothing) / (total_starts + smoothing * len(self.states))
            for next_tag in self.states:
                self.trans_probs[tag][next_tag] = (transition_counts[tag][next_tag] + smoothing) / (tag_counts[tag] + smoothing * len(self.states))
            for word in self.vocab:
                self.emit_probs[tag][word] = (emission_counts[tag][word] + smoothing) / (tag_counts[tag] + smoothing * len(self.vocab))

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        for tag in self.states:
            V[0][tag] = self.start_probs.get(tag, 1e-6) * self.emit_probs[tag].get(sentence[0], 1e-6)
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}
            for curr in self.states:
                (prob, prev_state) = max(
                    (
                        V[t - 1][prev_tag] *
                        self.trans_probs[prev_tag].get(curr, 1e-6) *
                        self.emit_probs[curr].get(sentence[t], 1e-6),
                        prev_tag
                    )
                    for prev_tag in self.states
                )
                V[t][curr] = prob
                new_path[curr] = path[prev_state] + [curr]
            path = new_path

        final_state = max(self.states, key=lambda tag: V[-1].get(tag, 0))
        return path[final_state]
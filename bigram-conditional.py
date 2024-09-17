from collections import defaultdict

# Sample text
text = "Mary had a little lamb, Its fleece was white as snow; and everywhere that Mary went, the lamb was sure to go."

# Tokenize the text into words (simple splitting and lowercasing)
words = text.lower().replace('.', '').split()

# Create bigrams (n=2)
bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]

# Calculate frequency of each bigram
bigram_freq = defaultdict(int)
for bigram in bigrams:
    bigram_freq[bigram] += 1

# Calculate frequency of each word
word_freq = defaultdict(int)
for word in words:
    word_freq[word] += 1

# Create a dictionary to hold the probability of each bigram
bigram_probabilities = defaultdict(dict)

# Calculate the conditional probability P(w2 | w1) = count(w1, w2) / count(w1)
for (w1, w2), freq in bigram_freq.items():
    bigram_probabilities[w1][w2] = freq / word_freq[w1]

# Function to predict the next word based on the current word
def predict_next_word(current_word):
    if current_word in bigram_probabilities:
        next_word_candidates = bigram_probabilities[current_word]
        # Choose the word with the highest probability
        next_word = max(next_word_candidates, key=next_word_candidates.get)
        return next_word
    else:
        return None

# Example predictions
current_word = 'the'
next_word = predict_next_word(current_word)
print(f"The next word after '{current_word}' is likely to be '{next_word}'.")

current_word = 'Mary'
next_word = predict_next_word(current_word)
print(f"The next word after '{current_word}' is likely to be '{next_word}'.")

#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np

# Load the dataset
with open("business-names.txt", "r") as file:
    data = file.read()

# Create a set of unique characters in the dataset
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("Data has %d characters, %d unique." % (data_size, vocab_size))

# Create dictionaries to map characters to indices and vice versa
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25    # number of steps to unroll the RNN for
learning_rate = 1e-1

# Model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))                        # hidden bias
by = np.zeros((vocab_size, 1))                         # output bias

# Loss function
def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # Forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t],0])

    # Backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

# Sample a sequence of characters from the model
def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

# Training loop
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

# Stopping condition
training = True
max_iterations = 1000  # Set maximum number of iterations

while training:
    # Prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # Sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    # Forward and backward pass
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # Print progress
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))

    # Perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter 

    # Check stopping condition
    if n >= max_iterations:
        training = False
        print("Training stopped after reaching maximum iterations.")
        
# Generate and print samples
sample_size = 5
for _ in range(sample_size):
    sample_ix = sample(hprev, inputs[0], 100)
    generated_text = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('Generated Sample:', generated_text)


# In[36]:


#Evaluation Methods

# Split data into training and validation sets
split = int(0.8 * len(data))
train_data, val_data = data[:split], data[split:]

# Evaluate on validation set
val_inputs = [char_to_ix[ch] for ch in val_data[:-1]]
val_targets = [char_to_ix[ch] for ch in val_data[1:]]
val_loss, _, _, _, _, _, _ = lossFun(val_inputs, val_targets, np.zeros((hidden_size,1)))

print('Validation Loss:', val_loss)


from Levenshtein import distance

# Generate sample
sample_ix = sample(hprev, inputs[0], 100)
generated_text = ''.join(ix_to_char[ix] for ix in sample_ix)

# Calculate Levenshtein distance
lev_distance = distance(generated_text, data[:len(generated_text)])

print('Levenshtein Distance:', lev_distance)


# Calculate perplexity
test_inputs = [char_to_ix[ch] for ch in data[:1000]]
test_targets = [char_to_ix[ch] for ch in data[1:1001]]
test_loss, _, _, _, _, _, _ = lossFun(test_inputs, test_targets, np.zeros((hidden_size,1)))
perplexity = np.exp(test_loss / len(test_inputs))

print('Perplexity:', perplexity)


# In[ ]:





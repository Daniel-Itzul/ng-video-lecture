# About this repo
This repository is a clone and little refactoring of [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) by Andrej Karpathy.

# Contents
It contains the following files:
- Bigram model based on a simple script utilizing conditional probabilities.
- Bigram model created by leveraging neural networks, a bit of an overkill, but a great way to deep dive in some basic concepts of Pytorch to understand better what the GPT model is doing.
- A sample GPT Model with all the basic building blocks of the transformer architecture.
- Training data (an extract of the full works of Shakespeare)
- Outputs: The output folder contains a sample of the type of text the model can generate with the current hyperparameters, of course, increasing the block size, adding more heads of attention, etc. would render better results but a GPU is needed in order to train with these improvements.

# Changes

The main changes I made was to modularize the functions for data extraction and training, and changing the hyperparameters to work on CPUs.

# Notes of the original repo

# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT

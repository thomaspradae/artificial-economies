# Stage 5 Model Exchange

This directory contains the Stage 5 bridge between the released DeepMind Torch7
Breakout network and the PyTorch Atari `QNetwork`.

The exchange is intentionally narrow:

- DeepMind creates the reference `convnet_atari3` model.
- Each convolution and linear layer is exported as raw little-endian `float32`
  weights and biases plus a JSON manifest.
- PyTorch imports those tensors into the real Atari `network.QNetwork`.
- The Stage 5 learner traces then compare forward values, Bellman targets,
  loss/gradient behavior, and one optimizer update from the same frozen
  minibatch.

The exchange does not resolve the Stage 2 resize issue. Stage 5 starts from
canonical DeepMind-produced `84x84` frames from Stage 4.

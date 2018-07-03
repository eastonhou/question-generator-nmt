1. requirements
VisualStudio Code + PyTorch 0.4.0

2. preparing data
(1) download and extract dureader dataset, put following files into ./data folder.
./data
  zhidao.train.json
  zhidao.dev.json
(2) run profile Python: preprocess

3. Training
run profile Python: train.
To run with teacher forcing (default), you don't need to specify arguments.
To run with generative adversarial training, add following argument.
  "args": [
    "-using_gan", "1"
  ]

4. Evaluating
run profile Python: evaluate.
PS: since dureader test set doen't have labels, validation set is used to compute the accuracy.
definition of accuracy: #correct words/#total words.

5. Baseline
42 iterations generative adverserial network: 43.29
44 iterations teaching forcing: evaluating 42.87
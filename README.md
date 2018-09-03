
# CORnet: Modeling Core Object Recognition

## What is this?

A family of simple yet powerful deep neural networks for visual neuroscience. What makes CORnets useful:

- Simple
- Lightweight
- Recurrent
- State-of-the-art on predicting primate neural and behavioral responses [Brain-Score](http://brain-score.org)

[read more...](#longer-motivation)

*Brought to you with <img src="resources/brain.png" width="25px"/> from [DiCarlo Lab](http://dicarlolab.mit.edu) @ [MIT](https://mit.edu).*

## Available Models

| Name     | Description                                                              |
| -------- | ------------------------------------------------------------------------ |
| CORnet-Z | Our smallest, fastest model. Good neural fits                            |
| CORnet-R | Recurrent version of CORnet-Z. Better than CORnet-Z + recurrent but slow |
| CORnet-S | CORnet-R with ResNet-like blocks. Best overall but slow to train         |


## Quick Start

### Want to test on your own images?

`python run.py test --restore_path <path to model weights> - --model S --data_path <path to your image folder>`

### Want to train on ImageNet?

1. [Get ImageNet](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) if you don't have it yet. If you do, note that validation images need to be put in separate folders, just like train data. Follow the instructions in that link to do so easily.

2. `python run.py train - --model Z`. On a single Titan X, it will train for about 14 hours.


# Details

## Longer Motivation

Deep neural networks have been established as the best candidate models for understanding visual processing in primate ventral visual pathway. Over past five years, these models evolved from simple feedforward eight-layer AlexNet architecture to extremely deep and intraconnected NASNet architectures, demonstrating increasingly better object categorization performance. However, from a neuroscientist point of view, the mapping between such very deep architectures and a relatively shallow hierarchy in the ventral visual pathway is puzzling. Morever, these state-of-the-art models lack recurrent dynamics that are characteristic to real biological systems. Here we describe our ongoing efforts to develop a CORnet family of architecures that would address these limitations. Starting from a basic four layer model, by leveraging the power of recurrent connectivity we gradually construct architectures that rival the best models in terms how brain-like they are. Our current best architecture, COR-Skip, outperforms all other models on Brain-Score, a composite benchmark for comparing models to the brain.

Read more: [Kubilius\*, Schrimpf\*, et al. (biorxiv, 2018)]

## Requirements

- Python 3.6+
- PyTorch 0.4.1+
- numpy
- pandas
- tqdm
- fire

## Citation

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D.L.K., DiCarlo, J.J. (2018) CORnet: Modeling Core Object Recognition. biorxiv.

## License

GNU GPL 3+


# FAQ

- Is CORnet-S *the* model of vision?

No. This is a constant work in progress. We display here our best current models for core object recognition but these models are constantly evolving.

- Why not "CoreNet"?

COR = Core Object Recognition. Also, CORnet has a nice connection to "cortex".

- My model is better than CORnet. Can I place it in this repository?

Exciting. Find a nice name for it and submit it to [Brain-Score.org](http://brain-score.org). This repository is only for CORnet family of models, while Brain-Score is a great place to show your model to the world and link to its own repository.

- Are hyperparameters arbitrary?

No. We tried many architectures and these are the ones that worked best. However, an exhaustive search has not been done. Simpler yet equally good models might exist, as well as more complicated but more predictive models.

- Why do you use classes for defining everything? Aren't functions enough?

Classes allow packaging functions into a single object, providing a good code organization.

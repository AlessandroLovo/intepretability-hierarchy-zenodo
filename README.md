# Code for the paper "Interpretability of prediction in a hierarchy of Machine Learning models for extreme heatwaves"

## Useful links

[repo of manuscript](https://github.com/amaurylancelin/Interpretability-heatwaves-paper)
[repo of supplementary material](https://github.com/amaurylancelin/SUPMAT-Interpretability-heatwaves-paper)

Preprint of paper: 

## Setup

To be able to run the notebooks in this repository, you need to clone the [Climate-Learning](https://github.com/georgemilosh/Climate-Learning) repository.

To do put yoursel in the same directory of this file and run

```bash
git clone --recursive https://github.com/georgemilosh/Climate-Learning.git
```

Reproducing the figures will use only the submodule [general_purpose](https://github.com/AlessandroLovo/general_purpose), but training the neural networks and visualizing them needs the full Climate-Learning framework.

## Contents of this repo

This repo contains data and notebooks to reproduce the figures and tables presented in our paper. The best way to navigate it is through its notebooks: inside each of them you'll find some explanatory markdown text.

### List of notebooks

- Data normalization procedure [here](interpretability-hierarchy/misc.ipynb)
- Performance of the hierarachy [here](interpretability-hierarchy/performance.ipynb)
- Interpretability
    - GA and IINN [here](interpretability-hierarchy/interpret_GA-IINN.ipynb)
    - CNN
        - Expected Gradient Feature Importance maps [here](interpretability-hierarchy/interpret_CNN_EGFI.ipynb)
        - Optimal Input Maps [here](interpretability-hierarchy/CNN-optimal-input/optimal_input.ipynb)
    - Scatnet [here](interpretability-hierarchy/interpret_ScatNet.ipynb)


### Debugging help

- Test if you can properly load our neural networks [here](interpretability-hierarchy/test_load_model.ipynb)
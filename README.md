# Deep Convolutional Generative Adversarial Network

PyTorch implementation of a DCGAN, based on the [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

### Training Workflow:
- Add an image dataset to the workspace
- Set up image dimensions in `config.py`
- Train models in `train.ipynb`
- Perform inferences in `inference.py` (Use the saving methods from `train_butterfly.ipynb`. They are NOT part of the PyTorch tutorial, which `train.ipynb` adheres to.)

The existing `train.ipynb` trains on the same celebA dataset as the PyTorch tutorial. Take a look at `train_butterfly.ipynb` as a step towards generalizing the procedure to other datasets.

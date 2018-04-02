# Sample Convolutional NN to classify chest X-Rays

2 classification examples on 150GB of data (in the reference link).
Images are 1024x1024
Make sure to change the directory where the source images are and the diagnoses(label) we're looking for.

WIP: exporting the model and getting the Golang piece to run inference - Doesn't seem to work on Windows
## Requirements
- Python 3.5+
- Tensorflow (compiled with CUDA 9.0 and CuDNN 7.0.5)
- Pandas, PIL, pydot, [graphviz](https://graphviz.gitlab.io)

# 1. Binary Classification
- classifies whether or not there is a problem identified in the scan
- Tested on a few diagnoses e.g. Cardiomegaly got ~95% accuracy on validation set

Can be run in the Jupyter Notebook on `chestrays-keras-binary-classification.ipynb`

It creates the folders: 
- `./train/NoFinding` and `./test/NoFinding` -- the program will copy scans identified with no findings here
- `./train/<label>` and `./test/<label>` -- copy of scans identified by label e.g. Cardiomegaly

Ensure there are only ever 2 subfolders in ./train and ./test.

# 2. Categorical Classification
- 188 categories with a small sample size. This needs to be cleaned up and the number of samples changed to reflect the entire data set
- Setup to use the keras image data preprocessing lib so that it can auto rotate and augment the images for an increased data set size
- This one needs tweaking for an increase in accuracy. It's more just sample code.

This example can be run without the Jupyter Notebook as it can cause a kernel crash with:
```bash
python chestrays-categorical.py
```

# Data set and article reference:
https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
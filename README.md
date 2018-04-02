# Sample Convolutional NN to classify chest X-Rays & Run inference in Golang

2 classification examples on 150GB of data (in the reference link).
Images are 1024x1024
Make sure to change the directory where the source images are and the diagnoses(label) we're looking for.

Different input and output tensor names can be generated while running the example. 
Make sure to display all the node names and find the input layer and output layer to run inference in Go.

This can be done with:
```python
[n.name for n in tf.get_default_graph().as_graph_def().node]
```

You should see this kind of output in Go if the model was run successfully:
```bash
(ML) tony@tony-nuc:$GOPATH/src/github.com/serinth/chestrays-ml-classification$ go run main.go
2018-04-02 20:30:51.905087: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-04-02 20:30:51.905281: I tensorflow/cc/saved_model/loader.cc:240] Loading SavedModel with tags: { myTag }; from: myModel
2018-04-02 20:30:51.913855: I tensorflow/cc/saved_model/loader.cc:159] Restoring SavedModel bundle.
2018-04-02 20:30:52.121236: I tensorflow/cc/saved_model/loader.cc:194] Running LegacyInitOp on SavedModel bundle.
2018-04-02 20:30:52.122132: I tensorflow/cc/saved_model/loader.cc:289] SavedModel load for tags { myTag }; Status: success. Took 216855 microseconds.
Result value: [[0.5441803]] 
```

`myModel` is the folder with the binary protobuf model saved
`myTag` is the TF tagged model


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
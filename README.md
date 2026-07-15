This repository contains python scripts for training TASE's ML-based path oracles using tensorflow.

The scripts take a comma-delimited file containing features as inputs, process the data, and output saved models for performing inferences as path oracles.  

Particularly large training data sets (i.e., data sets which cannot fully fit in RAM) require additional preprocessing to serialize the processed features into datasets on disk.  At training time, tensorflow then lazily streams data from these large processed data sets on disk to the GPU as needed per epoch/batch.

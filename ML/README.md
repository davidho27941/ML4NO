Data: Store on gpu 140.114.94.240
n1000000_0716.npz -- 3 sigma gaussian on theta12, theta13, theta23, and sdm, deltacp half on 0 and 180
n1000000_0728.npz -- flat on theta12, theta13, sdm, theta23 half on 45 and else flat, deltacp half on 0 and 180
n1000000_0803.npz -- fixed theta12 and sdm, flat on theta13, theta23 half on 45 and else flat, deltacp half on 0 and 180 (the preferred dataset for classification)
n1000000_0804_all_flat.npz -- fixed theta12 and sdm, flat on theta13, theta23, delta (the preferred dataset for regression)
NuFit_IO.npz -- the spectrum created by NuFit inverse ordering parameters
NuFit_NO.npz -- the spectrum created by NuFit normal ordering parameters

largerthan files: the index of the bin value for both IO and NO larger than the given value
    NIO_largerthan100_index.npy
    NIO_largerthan1000_index.npy
    NIO_largerthan500_index.npy

Feature files: A python notebook to inspect the features of the input data.
The name of the feature, the histogram of the feature, the correlation matrix of the ve_dune

Regression folder:

Feature -- to get understanding the features of original data
models -- to check the models created (the siffix represent the data used to train the model)
ak-delta -- build ak model on delta
ak-theta23 -- build ak model on theta23
ak -- build ak model on multiple parameters together
stat -- generate poisson bins and get statistics

Classification:

ak -- build ak model on all 12 classes (octant = {-1, 0, 1}, cpv = {0, 1}, mo = {-1, 1})
stat -- generate poisson bins and get statistics
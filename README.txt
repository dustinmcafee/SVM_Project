./data.py			splits input/vowel/vowel-context.data into training and testing dataset
./coarse.py kernel		uses the specified kernel (and hyper-parameters defined in the write-up for coarse grain search), prints K-fold mean accuracy/precision for each set of hyper-parameters
./fine.py			uses the radial basis kernel (and hyper-parameters defined in the write-up for fine grain search), prints K-fold mean accuracy/precision for each set of hyper-parameters
./test.py			uses the radial basis kernel (and hyper-parameters gamma = 5, C = 2, as defined in the write-up), prints confusion matrix performance metrics along with accuracy/precision

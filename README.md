# MatChain: Simple, Flexible, Efficient

MatChain is an experimental package designed for record linkage. Record linkage is the process of matching records that correspond to the same real-world entity in two or more datasets. This process typically includes several steps, such as blocking and the final matching decision, with a wide range of methods available, including probabilistic, rule-based, and machine learning approaches.

MatChain was created with three core objectives in mind: simplicity, flexibility, and efficiency. It focuses on unsupervised approaches to minimize manual efforts, allows for customization of matching steps, and offers fast and resource-efficient implementations.

MatChain leverages libraries like Pandas, NumPy, and SciPy for [vectorized data handling](https://ipython-books.github.io/45-understanding-the-internals-of-numpy-to-avoid-unnecessary-array-copying/), [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing), and support for [sparse matrices](https://docs.scipy.org/doc/scipy/tutorial/sparse.html).
It also utilizes scikit-learn and SentenceTransformers to convert strings into sparse vectors and dense vectors, respectively. This allows to perform blocking as approximate nearest neighbour search in the resulting set of vectors utilizing libraries like [NMSLIB](https://github.com/nmslib/nmslib) and [Faiss](https://github.com/facebookresearch/faiss).

The currently published version of MatChain exclusively provides [AutoCal](https://como.ceb.cam.ac.uk/preprints/293/) as the matching algorithm. AutoCal is an unsupervised method initially designed for instance matching with [Ontomatch](https://github.com/cambridge-cares/TheWorldAvatar/tree/main/Agents/OntoMatchAgent) in the context of [The World Avatar](https://theworldavatar.io/).
MatChain's implementation is highly efficient and allows
for the combination of AutoCal with various procedures for blocking and computing similarity scores.

## Installation

MatChain requires Python 3.8 or higher and can be installed with pip:

``` console
pip install matchain
```


## Basic Example Using the API

In this example, we demonstrate how to match two datasets, denoted as A and B, based on columns with the same names: "year," "title," "authors," and "venue." You can run this example in the accompanying notebook [run_matchain_api.ipynb](https://github.com/ae3000/matchain/blob/main/notebooks/run_matchain_api.ipynb), which provides a detailed explanation of MatChain's API, including how to specify parameters.

First, we read the data and initialize an instance of the class ```MatChain``` using Pandas' dataframes.

``` python
data_dir = './data/Structured/DBLP-ACM'
dfa = pd.read_csv(f'{data_dir}/tableA.csv')
dfb = pd.read_csv(f'{data_dir}/tableB.csv')

mat = matchain.api.MatChain(dfa, dfb)
```

Next, we specify one or more similarity functions for each matching column by the ```property``` method. These similarity functions calculate scores between 0 and 1 for pairs of column values. In this example, we use ```equal``` for the integer-valued "year" column, which returns 1 if two years are equal and 0 otherwise. For each of the remaining string-valued columns, we apply ```shingle_tfidf``` to generate a sparse vector for each string based on its shingles (n-grams on the character level) and compute the cosine similarity between the sparse vectors for pairs of strings:

``` python
mat.property('year', simfct='equal')
mat.property('title', simfct='shingle_tfidf')
mat.property('authors', simfct='shingle_tfidf')
mat.property('venue', simfct='shingle_tfidf')
```

As the total number of record pairs grows with the product of the record sizes in datasets A and B, classifying each pair as matching or non-matching can be computationally expensive, especially for large datasets. Blocking effectively reduces the number of pairs while only discarding a small fraction of true matching pairs. The following line specifies three columns to use for blocking. By default, MatChain utilizes the library [sparsedottopn](https://github.com/ing-bank/sparse_dot_topn) to perform blocking by conducting a nearest neighbor search on the same shingle vectors mentioned earlier:

``` python
mat.blocking(blocking_props=['title', 'authors', 'venue'])
```

Finally, we call ```autocal``` to execute the matching algorithm AutoCal and ```predict``` to get the predicted matching pairs:

``` python
mat.autocal()
predicted_matches = mat.predict()
```


## Configuration File

While the example above demonstrates how to use MatChain's API to match two datasets, an alternative and streamlined approach is to utilize a configuration file. This method allows us to specify datasets, matching chains, and parameters in a separate file:

``` console
python matchain --config ./config/mccommands.yaml
```

For more detailed information about configuration options, run the notebook [run_matchain_config.ipynb](https://github.com/ae3000/matchain/blob/main/notebooks/run_matchain_config.ipynb).



## Datasets

The data subdirectory includes pairs of example datasets and ground truth data for evaluating MatChain's performance. These datasets cover various domains, including restaurant, bibliography, product, and powerplants. Specifically, four of them originate from [this paper](https://dbs.uni-leipzig.de/files/research/publications/2010-9/pdf/EvaluationOfEntityResolutionApproaches_vldb2010_CameraReady.pdf) and were downloaded from the [DeepMatcher Data Repository](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md). Two additional dataset pairs are related to the powerplants domain and were originally used for [AutoCal](https://como.ceb.cam.ac.uk/preprints/293/).

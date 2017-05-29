pidpy
=====

pidpy is a Python package for partial information decomposition.
Specifically, it can be used to compute the pure terms of the decomposition
for an arbitrary number of source variables using the original proposal
of Williams & Beer [1].



Installation
------------
After cloning or downloading the repository, do

```
cd pidpy
```

Then you can install normally with

```
$ python setup.py install
```

or

```
$ pip install .
```

However, it is recommended to install with the _develop_ option, with

```
$ python setup.py develop
```

or

```
$ pip install -e .
```


Tests
-----
From the root directory, execute `nosetests`.

Quickstart
-----------
pidpy supports any number of feature variables with binary values
(taking on either `0` or `1`). For variables taking on integer non-binary
variables, support is limited to triplets.

In general, a good rule of thumb is to make sure that
`n_variables < log2(n_samples / (3*n_labels))`.

The main class `PIDCalculator` can be instantiated with a feature array `X` of
shape `(n_samples, n_variables)` and an array of labels `y` of shape `(n_samples,)`.

```python
from pidpy import PIDCalculator
pid = PIDCalculator(X,y)
```

Then the pure synergy, redundancy, and unique terms of the information
decomposition can be computed, together with the full mutual information (values
given in bits).

```python
pid.synergy()
pid.redundancy()
pid.unique()
pid.mutual()
````

The information values can also be debiased with shuffled surrogates, where
the parameter `n` specifies the number of surrogates to be used.

```python
pid.synergy(debiased = True, n = 50)
pid.redundancy(debiased = True, n = 50)
pid.unique(debiased = True, n = 50
pid.mutual(debiased = True, n = 50)
```
These methods return a tuple containing the original value debiased by
subtracting from it the mean of the surrogates, and the standard deviation
of the surrogates.

Lastly, the `decomposition` method can be used as a shortcut to get a full
picture of the partial information decomposition.

```python
synergy, redundancy, unique, mutual = pid.decomposition()
````
For more detailed information, see the documentation. To obtain a p-value
for every information theoretic quantity (computed as the fraction of
surrogates for which the information theoretic quantity is higher
than the observed one), you can do:

```python
decomposition, p_values = pid.decomposition(debiased = True, n = 50,
                                            return_std_surrogates = False,
                                            test_significance = True)
````
In this case, an additional tuple is returned containing a p-value
for every term.

Building the documentation
-----------
To build the docs, you need to have Sphinx installed, together with the
 `numpydoc` Sphinx extension (`pip install numpydoc`).
To build the documentation in html format:

```
cd docs
make html
```

Or

```
sphinx-build -b html ./source ./build/html
```

Then in `build/html` you can open `api.html` to begin reading
the documentation.

[1] Williams, P. L., & Beer, R. D. (2010).
Nonnegative decomposition of multivariate information.
arXiv preprint arXiv:1004.2515.
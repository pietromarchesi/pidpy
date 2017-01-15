pidpy
=====

pidpy is a Python package for partial information decomposition.
It computes

Installation
------------

Tests
-----
From the root directory, execute `nosetests`.

Quickstart
-----------
pidpy supports any number of feature variables with binary values
(taking on either `0` or `1`). For variables taking on integer non-binary
variables, support is limited to triplets.

In general, a good rule of thumb is to make sure that
`nvariables < log2(nsamples / (3*nlabels))`.

The main class `PIDCalculator` can be instantiated with a feature array `X` of
shape `(nsamples, nvariables)` and an array of labels `y` of shape `(nsamples,)`.

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
pid.debiased_synergy(n = 50)
pid.debiased_redundancy(n = 50)
pid.debiased_unique(n = 50
pid.debiased_mutual(n = 50)
```
These methods return a tuple containing the original value debiased by
subtracting from it the mean of the surrogates, and the standard deviation
of the surrogates.

Lastly, the `decomposition` method can be used as a shortcut to get a full
picture of the partial information decomposition.

```python
synergy, redundancy, unique, mutual = pid.decomposition()
````

By default, this function computes debiased values. To avoid the debiasing, pass
`debiased = False`. Synergy, redundancy, and unique terms are returned by
 default as a percentage of total mutual information. To obtain information
 values in bits, pass `as_percentage = False`. Unique information
  is returned by default as the sum of all the individual unique information
  contributions of the different variables. To return an array containing
  the individual contributions, pass `return_total_unique = True`.
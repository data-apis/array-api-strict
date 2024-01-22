array-api-strict
================

A strict, minimal implementation of the [Python array
API](https://data-apis.org/array-api/latest/)

Previously this implementation was available as `numpy.array_api`, but it was
moved to a separate package for NumPy 2.0.

Note: the history of this repo prior to commit
fbefd42e4d11e9be20e0a4785f2619fc1aef1e7c was generated automatically
from the numpy git history, using the following
[git-filter-repo](https://github.com/newren/git-filter-repo) command:

```
git_filter_repo.py --path numpy/array_api/ --path-rename numpy/array_api:array_api_strict --replace-text <(echo -e "numpy.array_api==>array_api_strict\nfrom ..core==>from numpy.core\nfrom .._core==>from numpy._core\nfrom ..linalg==>from numpy.linalg\nfrom numpy import array_api==>import array_api_strict") --commit-callback 'commit.message = commit.message.rstrip() + b"\n\nOriginal NumPy Commit: " + commit.original_id'
```

# Hilbert-Schmidt Independence Criterion (HSIC)

Python version of the original MATLAB code of [Hilbert-Schmidt Independence Criterion](http://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf) (HSIC).

## Prerequisites
* numpy
* scipy

We tested the code using **Anaconda 4.3.0 64-bit for python 2.7** on windows.

## Apply on your data

### Usage

Import HSIC using

```
from HSIC import hsic_gam
```

Apply HSIC on your data
```
testStat, thresh = hsic_gam(x, y, alph = 0.05)
```

### Description

Input of function `hsic_gam()`

| Argument  | Description  |
|---|---|
|x | Data of the first variable. `(n, dim_x)` numpy array.|
|y | Data of the second variable. `(n, dim_y)` numpy array.|
|alph | level of the test |

Output of function `hsic_gam()`

| Argument  | Description  |
|---|---|
|testStat  |test threshold for level alpha test|
|thresh| test statistic|

### Independence test result
- If **testStat < thresh**, `x` and `y` are independent.
- If **testStat > thresh**, `x` and `y` are not independent.

## Authors

* **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/HSIC/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

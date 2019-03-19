---
layout: post
excerpt: With Python and R code
images:
  - url: /assets/matrix_decomposition.png
---

### Introduction to Matrix Decomposition

**In the mathematical discipline of linear algebra, a matrix decomposition or matrix factorization is a factorization of a matrix into a product of matrices.**

The decomposed matrix are used where the direct calculation can be tedious or subject to underflow or overflow errors.
There are more than 30 matrix decomposition techniques in practice, In this blog i will try to shed some light on 5 most important techiques used in datascience. Each decomposition techniques severs some purpose depending on the principle in which they are based on. Following are the modern day applications of Matrix Decomposition:

* Solving system of linear equations
* Data Compression
* Collaborative Filtering ( Recommender Systems )
* Topic Modelling
* Noise Filtering
* Image processing and compression

Note: Modern day Recommendation system particular uses different algorithms like Alternating Least Square, Non Matrix Factorization and Singular Value Decomposition 

Below are the algorithms that we will discuss in this post.

1. LU Decomposition
2. QR Decomposition
3. Cholesky Decomposition
4. Spectral Decomposition or Principal Component Analysis
5. Singular Value Decomposition

### LU Decomposition

LU technique decomposes a matrix into an upper, a lower triangular matrix of same size as of the original matrix. It works only for the square matrix and not all the matrix can be decomposed using LU technique. For a non singular matrix A, the LU decomposition is not unique. There can be more than one LU decomposition for the same non singular matrix. The reason for non-unique solution is because of the elementary transformation while solving for L and U. To find out unique solution for LU, constraints should be placed on LU matrix.

<p>
  <script src="https://gist.github.com/karthickrajas/9f5feb0d674283426a99ce2c1e9c240c.js"></script>
</p>

##### Note
The LU decomposition is found iteratively and can't be successful for all the matrices easily. Hence a better/more stable way to solve the problem is called as LU decomposition with partial pivoting, which is built in python.

<p>
<script src="https://gist.github.com/karthickrajas/1d2e7d86f89a1de667cec10e41974a6d.js"></script>
</p>

##### Uses
* This method is highly useful in solving system of linear equations
* Finding the determinant ( Product of diagonal elements of upper and lower triangular matrix)
* solving for coefficients in linear regression

### QR Decomposition

If A is a  m×n  matrix with linearly independent columns, then A can be decomposed as QR, where Q is a m×n matrix whose columns form an orthonormal basis for the column space of A and R is an  non-singular upper triangular matrix. **QR decomposition is not limited to square matrix.**

```python
from numpy import array
from scipy.linalg import lu
# define a square matrix
A = array([[3,2], [9,3], [4,2]])
print(A)
# QR decomposition
Q, R = qr(A, 'complete')
print(Q)
print(R)
# reconstruct
B = Q.dot(R)
print(B)
```

``` r
x<-matrix(c(1,2,3, 2,5,4, 3,4,9),ncol=3,nrow=3)
qrstr <- qr(x)
Q<-qr.Q(qrstr)
R<-qr.R(qrstr)
```

##### Uses
* Solving Linear system of Equations
* Finding Eigen values of a matrix
* Least square Approximations

### Cholesky Decomposition

If A is a **real, symmetric and positive definite matrix** then there exists a unique lower triangular matrix L with positive diagonal element such that  A = L.LT
The Cholesky decomposition is for square symmetric matrices where all values are greater than zero (Positive definite matrix). For a Positive definite matrix all the eigen values with positive and non zero.
This can also be represented as a product of upper triangular matrix. A = UT.U

```python
from numpy import array
from numpy.linalg import cholesky
# define a 3x3 matrix
A = array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
print(A)
# Cholesky decomposition
L = cholesky(A)
print(L)
# reconstruct
B = L.dot(L.T)
print(B)
```

``` r
x<-matrix(c(4,2,-2, 2,10,2, -2,2,5),ncol=3,nrow=3)
cl<-chol(x)
```

##### Uses
* Cholesky can be used in the place of LU decomposition where the matrices are positive definite. 
* Cholesky is twice as efficient as LU decomposition.
* Cholesky is very useful since symmetric, positive definite matrices are very frequent in several fields. eg: Portfolio Optimization in Finance.

### Spectral Decomposition

Spectral Analysis is based on Eigen values of the matrix. Let A be a m × m real symmetric matrix. Then there exists an orthogonal matrix P such that (PT.A.P) = Lambda or (A = P.lambda.PT), where lambda is diagonal matrix. Spectral Decomposition can be explained with its application Principal Component Analysis. Principal Component Analysis is a popular dimensionality reduction method. PCA of a matrix A,nxm, simply projects each datapoints into a subspace with m or fewer columns, while retaining the variance explained by each variable.
PCA is a transformation on the data in to lesser dimensional space. This technique could also be employed when the data is highly co-related. 

```python
#libraries
import numpy as np
A = np.array([[0,2,1],[3,-1,2],[1,-1,1]])
print("rank(A) =", np.linalg.matrix_rank(A))
lambda, U = np.linalg.eig(A)
print("Eigen values:", lambda)
print("\n Eigen vectors :", U)
print(U*diag(lambda)*U.T)
```

``` r
X<-matrix(c('2,0,0,0,3,4,4,9'),ncol=3,nrow=3)
eigen(X) 
```

##### Uses:
* Principal Components Extraction
* Feature Extraction
* Dimensionality Reduction
* Visualizing Data points, word vectors (NLP)
* Factor Analysis

### Singular Value Decomposition

Singular value decomposition is similar to decomposition using eigen values, but its more generally applicable mainly because **spectral decomposition is only available for square matrix** . Every real matrix has a singular value decomposition, which is not the case with eigen value decomposition. In singular value decomposition we will write A as a product of three matrices:
A = U.D.VT

```python
import numpy as np
A = np.array([[14,2],[4,22],[16,13]])
U.D.VT = np.linalg.svd(A, full_matrices=False)
print("singular values : ", D)
print(" Left singular matrix :",U)
print(" Right singular matrix :", VT)
```

``` r
x<-matrix(c(14,2,4,22,16,13),ncol=3,nrow=3)
sv<-svd(x)
D<-sv$d
U<-sv$u
V<-sv$v
```

##### Uses
* More numerically stable Principal Components
* Image processing and Compression
* Multivariate Outliers detection
* Noise filtering
* Data reduction

#### References

* [Prof.Nishith Kumar](https://www.ime.unicamp.br/~cnaber/Matrix-Decomposition-and-Its-application-in-Statistics_NK.ppt)
* [Dr. Jason Brownlee](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)

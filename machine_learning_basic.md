# Machine Learning

## 1 Machine learning Overview

### 1.1 Supervised Learning

Give (input: x, output: y), learns from being given "right answers"

Type 1: Regression

​	Housing price prediction

​	straight line, fitting curve

Type 2: Classification: predict a small (finite) number of class / category

​	Breast cancer detection: benign 0 or malignant type 1, type 2, type 3... (良性或恶性)

​	If two or more import (age, tumor size), we find a boundary separate the benign and malignant ones.

### 1.2 Unsupervised Learning

Find something interesting om unlabeled data.

e.g. Clustering algorithm: tell the machine how to cluster (e.g. by finding features)

​	Google news

​	Find similar topics that users are interested in.

​	Grouping customers

​	Clustering by reasons that customers choose the same service.

Formal def.: Data only comes with inputs x, but not output labels y. Algorithm has to find **structure** in the data.

Type 1: Clustering

​	Group similar data points together.

Type 2: Anomaly detection (反常检测)

​	Find unusual data points.

Type 3: Dimensionality reduction

​	Compress data using fewer numbers.

## 2 Linear Regression with One Variable

### 2.1 Linear Regression Model

**Regression**: Any supervised learning model that predicts a number, is addressing a regression problem. Regression model predicts **numbers**. **Infinitely many possible outputs.** 

**Classification**: Classification model predicts categories. **Small number of possible outputs.**

**Terminology:**

Training set: Data used to train the model.

Notation:

$x$ = "input" variable / feature (input feature) (输入特征)

$y$ = "output" variable / "target" variable

$m$ = total number of training examples

$(x, y)$ = single training example

Because there are more than one examples, thus we use:

$(x^{(i)}, y^{(i)})$ = $i^{th}$ training example 

**Process:**

training set -> learning algorithm -> model $f$: a function

$x$: input (feature) -> model $f$ -> $\hat{y}$: prediction (the estimated value of y (the true value))

How to represent $f$?

If a straight line, $f_{w, b}(x) = f(x) \text{(simple notation)} = wx + b$

Sometimes, linear function is relatively easy to construct and show the relationship. Thus, here $f(x)$ is a **linear regression with one variable**, or **univariate (单变量的) linear regression**. **Linear regression means the prediction model is linear** instead of non-linear.

### 2.2 Cost function

Here we still focus on one variable linear regression model.

**Terminology:**

Model: $f(x) = wx + b$

$w, b$: parameters, coefficients, weights

intercept (截距): the value of y while $x = 0$

slope (斜率): here is the value of w

because $\hat{y}^{(i)} = f(x^{(i)}) = wx^{(i)} + b$, which is not $y$, so the target here is to find w, b s.t. $\hat{y}^{(i)}$ is close to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$.

**Cost function:** also called squared error cost function

error: $\hat{y}^{(i)} - y^{(i)}$

$m$ = number of training examples

function: $J(w, b) = \frac{1}{2m}\sum_{i = 1}^m(\hat{y}^{(i)} - y^{(i)})^2$ = $J(w, b) = \frac{1}{2m}\sum_{i = 1}^m(f(x^{(i)}) - y^{(i)})^2$

An $\frac{1}{2}$ before $1\over{m}$ is for easier calculation (differential) 

### 2.3 Cost Function Intuition

To simplified the model, here we give:

$f(x) = wx$, let b = 0

$J(w) = \frac{1}{2m}\sum_{i = 1}^m(f(x^{(i)}) - y^{(i)})^2$

So the goal here is:

${minimize}_{w}J(w)$

Consider $(1, 1), (2, 2), (3, 3)$

If we choose $w = 1$, then we can get $J(1) = 0$.

If we change the value of $w$, the graph on the left changes as well, and finally, If we gather all the points, we have the graph on the right, while each line on the left is corresponding to a point on the right.

![](files\1.png)

Goal of linear regression:

${minimize}_{w}J(w)$

General case:

${minimize}_{w, b}J(w, b)$

### 2.4 Visualizing the Cost Function










# TenserFlowNeuralNetwork
This a Self-learning repo for NN using tenser flow

# Tenser
A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes
Each tensor represents a partialy defined computation that will eventually produce a value.
Each tensor has a data type and a shape.
Data Types Include: float32, int32, string and others.
Shape: Represents the dimension of data.
### Rank/Degree of Tensors
Another word for rank is degree, these terms simply mean the number of dimensions involved in the tensor. What we created above is a tensor of rank 0, also known as a scalar.
```python
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
```
The rank of a tensor is direclty related to the deepest level of nested lists. You can see in the first example ["Test"] is a rank 1 tensor as the deepest level of nesting is 1. Where in the second example [["test", "ok"], ["test", "yes"]] is a rank 2 tensor as the deepest level of nesting is 2.
### Shape of Tensors
Now that we've talked about the rank of tensors it's time to talk about the shape. The shape of a tensor is simply the number of elements that exist in each dimension. TensorFlow will try to determine the shape of a tensor but sometimes it may be unknown.
To get the shape of a tensor we use the shape attribute.
```python
rank2_tensor.shape
```
Changing Shape
The number of elements of a tensor is the product of the sizes of all its shapes. There are often many shapes that have the same number of elements, making it convient to be able to change the shape of a tensor.
```python
tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,2]
                                                                             
# The numer of elements in the reshaped tensor MUST match the number in the original
```
```counsole
tf.Tensor(
[[[1. 1. 1.]
  [1. 1. 1.]]], shape=(1, 2, 3), dtype=float32)
tf.Tensor(
[[[1.]
  [1.]
  [1.]]

 [[1.]
  [1.]
  [1.]]], shape=(2, 3, 1), dtype=float32)
tf.Tensor(
[[1. 1.]
 [1. 1.]
 [1. 1.]], shape=(3, 2), dtype=float32)
```

basically the first number in reshape number of lists, second one is number of lists inside EACH list and the third number is number of vlaues in the list for negative value it calculate the needed number to put the values  we have 6 values so 6/3 is 2 
###  Ragged Tensors
A tensor with variable numbers of elements along some axis is called "ragged". Use tf.ragged.RaggedTensor for ragged data.A tf.RaggedTensor, shape: [4, None]
```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
print(ragged_tensor.shape)
```
```counsole
<tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
(4, None)
```
###  Sparse tensors
Sparse tensors store values by index in a memory-efficient manner.A tf.SparseTensor, shape: [3, 4]
```python
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))
```
```counsole
SparseTensor(indices=tf.Tensor(
[[0 0]
 [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 

tf.Tensor(
[[1 0 0 0]
 [0 0 2 0]
 [0 0 0 0]], shape=(3, 4), dtype=int32)
```
### Training
or this specific model data is going to be streamed into it in small batches of 32. This means we will not feed the entire dataset to our model at once, but simply small batches of entries. We will feed these batches to our model multiple times according to the number of epochs.The reason is because its slower to load all dtat at once.

An epoch is simply one stream of our entire dataset. The number of epochs we define is the amount of times our model will see the entire dataset. We use multiple epochs in hope that after seeing the same data multiple times the model will better determine how to estimate it.


###  TensorFlow Core Learning Algorithms
# Linear Regression
Linear regression is one of the most basic forms of machine learning and is used to predict numeric values.
Line of best fit refers to a line through a scatter plot of data points that best expresses the relationship between those points.
```math
y=mx+b
```
![image](https://github.com/Zyadsowilam/TenserFlowNeuralNetwork/assets/96208685/57480373-1ded-47a3-9671-81728d532833)
# Classification
classification is used to seperate data points into classes of different labels.Arranging or sorting objects into groups on the basis of a common property that they have.

1. Logistic Regression
Logistic regression is used for binary classification problems.

```math
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) 
```
where:
```math
- \(\mathbf{x}\) is the input feature vector.\text{is the input}
```
- \(\mathbf{w}\) is the weight vector.
- \(b\) is the bias term.
- \(\sigma(z)\) is the sigmoid function: \(\sigma(z) = \frac{1}{1 + e^{-z}}\).
```
The decision boundary is given by:
\[ \mathbf{w}^\top \mathbf{x} + b = 0 \]

The cost function to be minimized (log loss) is:
```math
\[ J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\mathbf{w}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\mathbf{w}(\mathbf{x}^{(i)})) \right] \]
```
where \( h_\mathbf{w}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) \) and \(m\) is the number of training examples.

 2. Decision Trees
Decision trees classify instances by sorting them down the tree from the root to some leaf node, which provides the classification.

The process involves splitting the dataset into subsets based on the feature that results in the most homogeneous subsets (purest).

**Impurity Measures:**
- **Gini Index:**
- ```math
  \[ Gini(D) = 1 - \sum_{i=1}^{c} p_i^2 \]
  ```
  where \( p_i \) is the probability of an element being classified for a particular class.

- **Entropy:**
 ```math
  \[ Entropy(D) = - \sum_{i=1}^{c} p_i \log_2(p_i) \]
```
The best split is chosen based on the highest information gain:
\[ \text{Information Gain} = Entropy(D) - \sum_{k=1}^{K} \frac{|D_k|}{|D|} Entropy(D_k) \]
where \( D_k \) is the subset created from the split.

 3. Support Vector Machines (SVM)
SVMs find the hyperplane that best separates the classes in the feature space.


For a binary classification:
```math
\[ f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b \]
```
The goal is to maximize the margin, which is the distance between the hyperplane and the closest points from either class (support vectors).

The optimization problem is:
\[ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \]
subject to:
\[ y^{(i)} (\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1, \; \forall i \]

For the non-linear case, kernel functions

\( K(\mathbf{x}_i, \mathbf{x}_j) \) are used to map input features into higher-dimensional spaces:
\[ f(\mathbf{x}) = \sum_{i=1}^{m} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \]
where \( \alpha_i \) are the Lagrange multipliers.

###  Credit to TechWithTim
https://www.youtube.com/watch?v=tPYj3fFJGjk

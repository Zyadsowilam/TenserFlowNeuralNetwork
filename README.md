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


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


# TensorFlow Core Learning Algorithms
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
- \mathbf{x}  \text{ is the input feature vector.}
```
```math
- \mathbf{w} \text{ is the weight vector.}
```
```math
- \mathbf{b} \text{ is the bias term.}
```
```math
- \sigma(z) \text{ is the sigmoid function: } \sigma(z) = \frac{1}{1 + e^{-z}}.
```
The decision boundary is given by:
```math
 \mathbf{w}^\top \mathbf{x} + b = 0
```
The cost function to be minimized (log loss) is:
```math
 J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\mathbf{w}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\mathbf{w}(\mathbf{x}^{(i)})) \right] 
```

```math
\text{ where } ( h_\mathbf{w}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) ) \text{ and  } \mathbf{m}  \text{ where is the number of training examples.}
```
 2. Decision Trees
Decision trees classify instances by sorting them down the tree from the root to some leaf node, which provides the classification.

The process involves splitting the dataset into subsets based on the feature that results in the most homogeneous subsets (purest).

**Impurity Measures:**
- **Gini Index:**
- ```math
  [ Gini(D) = 1 - \sum_{i=1}^{c} p_i^2 ]
  ```

```math
\text{ where } \mathbf{p_i}   \text{ is the probability of an element being classified for a particular class.  }
```
- **Entropy:**
 ```math
  [ Entropy(D) = - \sum_{i=1}^{c} p_i \log_2(p_i) ]
```
The best split is chosen based on the highest information gain:

 ```math
  [Information Gain = Entropy(D) - \sum_{k=1}^{K} \frac{|D_k|}{|D|} Entropy(D_k) ]
```
```math
\text{ where }   \mathbf{D_k}   \text{ is the subset created from the split.  }
```
 3. Support Vector Machines (SVM)
SVMs find the hyperplane that best separates the classes in the feature space.


For a binary classification:
```math
[ f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b ]
```
The goal is to maximize the margin, which is the distance between the hyperplane and the closest points from either class (support vectors).

The optimization problem is:
```math
[ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 ]
```

subject to:
```math
[ y^{(i)} (\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1, \; \forall i ]
```


For the non-linear case, kernel functions
```math
( K(\mathbf{x}_i, \mathbf{x}_j) )  
```
are used to map input features into higher-dimensional spaces:
```math
[ f(\mathbf{x}) = \sum_{i=1}^{m} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b ] \text{ where} ( \alpha_i ) \text{  are the Lagrange multipliers.} 
```
# Clustring
K-means clustering aims to partition \(n\) data points into \(k\) clusters, where each data point belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
![image](https://github.com/Zyadsowilam/TenserFlowNeuralNetwork/assets/96208685/f73cf803-1db0-4ce4-ba5a-ba7d953ab26d)




### K-means Clustering Algorithm

1. **Initialization:**
    $$\(k\) \text{ initial cluster centroids  raandomise} .$$
```math
 \text{ Select}  f(\mathbf{k})  \text{  initial cluster centroids} (\{\mathbf{\mu}_1, \mathbf{\mu}_2, \ldots, \mathbf{\mu}_k\})
```
2. **Assignment Step:**
   Assign each data point \(\mathbf{x}_i\) to the nearest cluster centroid:
   
```math
    [ c_i = \arg\min_{j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2 ] 

```
```math
 \text{ where}  (c_i)  \text{  is the index of the cluster centroid closest to} (\mathbf{x}_i).
```
3. **Update Step:**
   Update the cluster centroids by calculating the mean of all data points assigned to each cluster:
   
```math
 [ \mathbf{\mu}_j = \frac{1}{|C_j|} \sum_{\mathbf{x}_i \in C_j} \mathbf{x}_i ] 
```

```math
 \text{ where}  (c_j)  \text{   is the set of data points assigned to cluster }  (j) \text{and} (|C_j|)  \text{is the number of points in} (C_j).
```
4. **Repeat:**
   Repeat the assignment and update steps until convergence (i.e., when the assignments no longer change or the change in centroids falls below a threshold).

### Objective Function

K-means clustering aims to minimize the within-cluster sum of squares (WCSS), also known as inertia:
```math
[ J = \sum_{j=1}^{k} \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2 ]

```
- **Distance Calculation:**
 ```math
  [ \|\mathbf{x}_i - \mathbf{\mu}_j\|^2 = \sum_{d=1}^{D} (x_{i,d} - \mu_{j,d})^2 ]
```
  where (D) is the dimensionality of the data points.

- **Cluster Assignment:**
```math
  [ c_i = \arg\min_{j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2 ]
```
- **Centroid Update:**
```math
  [ \mathbf{\mu}_j = \frac{1}{|C_j|} \sum_{\mathbf{x}_i \in C_j} \mathbf{x}_i ]
```
- **Objective Function:**
```math
  [ J = \sum_{j=1}^{k} \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2 ]
```
By iteratively performing the assignment and update steps, K-means clustering minimizes the objective function, leading to well-defined clusters.
### Hidden Markov Models
A finite set of states, each of which is associated with a (generally multidimensional) probability distribution .
![image](https://github.com/Zyadsowilam/TenserFlowNeuralNetwork/assets/96208685/980243b6-1925-48d7-8a7b-ddbb5367f2e6)

### Key Components of HMMs

1. **States (\(S\)):**
   A set of (N) hidden states
   ```math
   (S = \{S_1, S_2, \ldots, S_N\})
   ```

3. **Observations (O):**

```math
 \text{ A sequence of }  (T)  \text{   observations }  (O = \{O_1, O_2, \ldots, O_T\}) \text{and} (|C_j|)  \text{where each observation comes from a finite set of symbols.}
```
4. **Transition Probabilities (\(A\)):**
   The probability of transitioning from one state to another:
```math
   [ A = \{a_{ij}\} \quad \text{where} \quad a_{ij} = P(S_{t+1} = S_j \mid S_t = S_i) ]
```
6. **Emission Probabilities (B):**
The probability of observing a symbol given a state:
```math
   
   [ B = \{b_j(o_t)\} \quad \text{where} \quad b_j(o_t) = P(O_t = o \mid S_t = S_j) ]
```
8. **Initial State Probabilities (\(\pi\)):**
The probability of starting in a particular state:
```math
   
   [ \pi = \{\pi_i\} \quad \text{where} \quad \pi_i = P(S_1 = S_i) ]
```
### Fundamental Problems in HMMs

 1. Evaluation (Forward Algorithm)

Given an HMM and an observation sequence \(O\), calculate the probability of the observation sequence:

**Forward Probability (alpha):**
```math
[ \alpha_t(j) = P(O_1, O_2, \ldots, O_t, S_t = S_j \mid \lambda) ]
```
Recurrence relations:
```math
[ \alpha_1(j) = \pi_j b_j(O_1) ]
```
```math

[ \alpha_{t+1}(j) = \left[ \sum_{i=1}^{N} \alpha_t(i) a_{ij} \right] b_j(O_{t+1}) ]
```
The probability of the observation sequence is:
```math
[ P(O \mid \lambda) = \sum_{j=1}^{N} \alpha_T(j) ]
```
 2. Decoding (Viterbi Algorithm)

Given an HMM and an observation sequence \(O\), find the most probable state sequence:

**Viterbi Variable (\(\delta\)):**
```math
[ \delta_t(j) = \max_{S_1, S_2, \ldots, S_{t-1}} P(S_1, S_2, \ldots, S_t = S_j, O_1, O_2, \ldots, O_t \mid \lambda) ]
```
Recurrence relations:
```math
[ \delta_1(j) = \pi_j b_j(O_1) ]
```
```math
[ \delta_{t+1}(j) = max_{i} [\delta_t(i) a_{ij}] b_j(O_{t+1}) ]
```
To retrieve the state sequence, backtracking is used:
```math
[ S_T^* = \arg max_j \delta_T(j) ]
```
```math
[ S_t^* = \psi_{t+1}(S_{t+1}^*) \quad \text{for} \quad t = T-1, T-2, \ldots, 1 ]
```
```math
\text{where} (\psi) \text{ stores the state with the highest probability at each step.}
```
3. Learning (Baum-Welch Algorithm / Expectation-Maximization)

Given an HMM and an observation sequence (O), adjust the model parameters to maximize the probability of the observation sequence:

**Forward Variable (alpha) and Backward Variable (beta):**
```math
[ \alpha_t(j) = P(O_1, O_2, \ldots, O_t, S_t = S_j \mid \lambda) ]
```
```math
[ \beta_t(i) = P(O_{t+1}, O_{t+2}, \ldots, O_T \mid S_t = S_i, \lambda) ]
```

**Re-estimation formulas:**
```math
[ \gamma_t(i) = P(S_t = S_i \mid O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)} ]
```
```math
[ \xi_t(i, j) = P(S_t = S_i, S_{t+1} = S_j \mid O, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(O_{t+1}) \beta_{t+1}(j)}{P(O \mid \lambda)} ]
```
**Updated parameters:**

```math

[ \pi_i = \gamma_1(i) ]
```
```math
[ a_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)} ]
```
```math
[ b_j(o_k) = \frac{\sum_{t=1}^{T} \delta(O_t = o_k) \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)} ]
```

```math
\text{Here} (\delta(\cdot))  \text{ is an indicator function that is 1 if} (O_t = o_k) \text{ and 0 otherwise.} 
```
# Neural Network
Please check the scratch NN for Mathmatics and concept of Algo

# Credit to TechWithTim
https://www.youtube.com/watch?v=tPYj3fFJGjk

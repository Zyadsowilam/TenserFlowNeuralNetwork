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
https://github.com/Zyadsowilam/ScratchNeuralNetwork
# Deep Computer Vision
Each convolutional neural network is made up of one or many convolutional layers. These layers are different than the dense layers we have seen previously. Their goal is to find patterns from within images that can be used to classify the image or parts of it. 
```math

[ (I * K)(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) \cdot K(m, n) ]
```
Here is a breakdown of the components in the equation:

- (I) is the input image or feature map.
- (K) is the kernel or filter.
- (i) and (j) are the coordinates of the output pixel.
- (m) and (n) are the coordinates within the kernel.
- (M) and (N) are the dimensions of the kernel.
```math
\text{The result of the convolution operation, } ( (I * K)(i, j) ), \text{ is a new feature map where each pixel is computed by summing the element-wise product of the kernel and the corresponding patch of the input image.}
```
In practice, a CNN convolutional layer may also include additional steps such as:

1. **Bias Addition**: A bias term \( b \) can be added to the result of the convolution:
```math
   [ (I * K)(i, j) = \left(\sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) \cdot K(m, n)\right) + b ]
```
2. **Activation Function**: An activation function (such as ReLU) can be applied to introduce non-linearity:
```math
   [ A(i, j) = f((I * K)(i, j)) ]
```
   where ( f ) is the activation function.

Putting it all together, the output of a convolutional layer at position \( (i, j) \) can be expressed as:
```math
[ A(i, j) = f\left(\left(\sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) \cdot K(m, n)\right) + b\right) ]
```
# Recurrent Neural Networks (RNN's)

### Simple RNN Equations
![alt text](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)
```math
\text{In a simple RNN, the hidden state } (h_t)  \text{ at time step } (t) \text{ is computed using the current input }  x_t \text{ and the hidden state from the previous time step }  h_{t-1} :
```


1. **Hidden State Update:**
```math
[
   h_t = \tanh(W_h x_t + U_h h_{t-1} + b_h)
   ]
```
  

   where:
   ```math

 - ( W_h ) \text{ is the weight matrix for the input }  ( x_t ).
```
   ```math

 - ( U_h ) \text{ is the weight matrix for the previous hidden state }  (  h_{t-1} ).
```
   ```math

 - ( b_h ) \text{ is the bias term. } 
```
   ```math

 - ( \tanh ) \text{ is the hyperbolic tangent activation function. }  
```


2. **Output:**

```math

 y_t = W_y h_t + b_y
```

   where:
  ```math

 - ( W_y ) \text{ is the weight matrix for the hidden state }  ( h_t ) \text{  to the output } .
```
   ```math

 - ( b_y ) \text{ is the bias term for the output. } 
```


### LSTM Equations

LSTM networks have a more complex structure with a memory cell ( c_t ) that helps in retaining long-term dependencies. They include gates to control the flow of information.
![image](https://github.com/Zyadsowilam/TenserFlowNeuralNetwork/assets/96208685/b76db2b5-4cf8-43bb-9c10-e7ef7a64b325)
1. **Forget Gate:**
```math

  [
   f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
   ]
```
 ```math
\text{where} (\sigma )  \text{ is the sigmoid function.} 
```

2. **Input Gate:**
 ```math
   [
   i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
   ]
```


3. **Candidate Memory Cell:**
 ```math
  [
   \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
   ]
```
 

4. **Memory Cell Update:**
 ```math
    [
   c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
   ]
```
  ```math
\text{where} (\odot )  \text{ denotes element-wise multiplication.} 
```

5. **Output Gate:**
 ```math
   [
   o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
   ]

```
  
6. **Hidden State Update:**
 ```math
   [
   h_t = o_t \odot \tanh(c_t)
   ]
```


### Summary of Parameters
  ```math
- ( W ) \text{matrices are weights for the current input} (x_t ) 
```
 ```math
- ( U ) \text{ matrices are weights for the previous hidden state } ( h_{t-1} ).
```
  ```math
- ( b ) \text{vectors are bias terms.} 
```
  ```math
- ( \sigma ) \text{is the sigmoid activation function.} 
```
  ```math
- ( \tanh )   \text{ is the hyperbolic tangent activation function.} 
```
  ```math
- ( \odot ) \text{denotes element-wise multiplication.} 
```

### Differences

- **Simple RNNs** use a single equation to update the hidden state, making them simpler but prone to vanishing gradient problems.
- **LSTMs** introduce a memory cell and gates to control information flow, helping to mitigate the vanishing gradient problem and better capture long-term dependencies.
# REINFORCMENT
  ```math
[ V(s) = \mathbb{E}[R_{t+1} + \gamma V(s_{t+1}) \mid s_t = s] ]
```
- \( \)  \( s \) to \( s_{t+1} \),
where:

```math
- (  V(s) ) \text{ is the value of state }   ( s )
```
```math
- (  \mathbb{E} ) \text{ denotes the expected value,  }  
```
```math
- (   R_{t+1} ) \text{ is the reward received after transitioning from state  }   ( s  ) \text{ to } ( s_{t+1} ),
```
```math
- (  \gamma ) \text{ is the discount factor, and }   
```
```math
- (   s_{t+1}  ) \text{ is the next state. }   
```

### Q-Learning 
The Q-learning update rule for the Q-value ( Q(s, a) ) is given by:
  ```math
[ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s, a) \right] ]
```

where:

 ```math
- (   Q(s, a)  ) \text{in state }   ( s )
```
```math
- (   \alpha  ) \text{ is the learning rate, }   
```
```math
- (   R_{t+1}  ) \text{ is the reward received after taking action  }   ( a ) \text{ in state  } ( s) 
```
```math
- (  \gamma ) \text{ is the discount factor, }   
```
```math
- (   s_{t+1}  ) \text{  is the next state, and }   
```
```math
- (  \max_{a'} Q(s_{t+1}, a')    ) \text{ is the maximum Q-value for the next state }   ( s_{t+1} ) \text{  over all possible actions }  ( a' ).
```
#### Updating Q-Values
The formula for updating the Q-Table after each action is as follows:
```math
 Q[state, action] = Q[state, action] + \alpha * (reward + \gamma * max(Q[newState, :]) - Q[state, action])
```
```math
- (  \alpha    ) \text{ stands for the **Learning Rate** }   
```
```math
- (  \gamma ) \text{  stands for the **Discount Factor** }  
```

# Credit to TechWithTim
https://www.youtube.com/watch?v=tPYj3fFJGjk

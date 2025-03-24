# Steam Game Recommender System
## Methodology
### 1. Problem Definition

The goal of the Steam Game Recommender System is to predict the playtime of users for different games based on their historical playtime data. The system leverages a collaborative filtering approach using matrix factorization with Singular Value Decomposition (SVD) and incorporates user and game biases to enhance prediction accuracy.

### 2. Data Representation

Let the playtime matrix $( R )$ represent the hours played by users for various games. Each entry $( r_{u,i} )$ in the matrix corresponds to the playtime of user $( u )$ on game $( i )$. Missing entries indicate games a user has not played.

### 3. Normalization

To account for user-specific playtime habits and registration duration, we normalize the playtime matrix $( R )$ before applying matrix factorization. This ensures that users with longer registration periods do not disproportionately influence the model.

Given:

- $` R `$ is the original playtime matrix (in hours).
- $` t_u `$ represents the time since user $` u `$ registered.
- $` \epsilon `$ is a small constant to prevent division by zero.

#### Normalization Process

1. **Normalize by registration duration**:  
   Each user's playtime is divided by the time since their registration:

```math
\tilde{r}_{u,i} = \frac{r_{u,i}}{60 \times t_u}
```
2. **Apply the natural logarithm**:  
   To reduce the impact of extreme values and stabilize variance, we take the natural logarithm of the normalized playtime:

```math
\tilde{r}_{u,i} = \log \left( \frac{r_{u,i}}{60 \times t_u} + \epsilon \right)
```

This transformation produces a normalized playtime matrix where values reflect playtime relative to the user's registration period, mitigating biases from longer account ages.

### 3. Matrix Factorization with Biases

We model the observed playtime using the following decomposition:

```math
\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i
```
Where:
- $` \hat{r}_{u,i} `$ is the predicted playtime.
- $` mu `$ is the global mean playtime.
- $` b_u `$ is the user bias.
- $` b_i `$ is the game bias.
- $` \mathbf{p}_u `$ and $` \mathbf{q}_i `$ are the latent factor vectors for user $` u `$ and game $` i `$ respectively.

### 4. Loss Function

We minimize the regularized squared error loss function to learn the parameters:

```math
L = \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda (||\mathbf{p}_u||^2 + ||\mathbf{q}_i||^2 + b_u^2 + b_i^2)
```

Where:
- $` \mathcal{K} `$ is the set of known playtime observations.
- $` \lambda `$ is the regularization parameter to prevent overfitting.

### 5. Singular Value Decomposition (SVD) Initialization

We initialize the latent matrices $` P `$ and $` Q `$ using the truncated SVD of the playtime matrix $` R `$:

```math
R \approx U \Sigma V^T
```

Where:
- $` U `$ is the user matrix.
- $` \Sigma `$ is a diagonal matrix of singular values.
- $` V `$ is the game matrix.

We set $` P = U \sqrt{\Sigma} `$ and $` Q = V \sqrt{\Sigma} `$ for initialization.

### 6. Stochastic Gradient Descent (SGD) Optimization

We optimize the parameters using Stochastic Gradient Descent. The update rules are as follows:

1. For each observed $` (u, i) `$ in $` \mathcal{K} `$:

```math
e_{u,i} = r_{u,i} - \hat{r}_{u,i}
```

2. Update biases:

```math
 b_u \leftarrow b_u + \eta (e_{u,i} - \lambda b_u)
```

```math
 b_i \leftarrow b_i + \eta (e_{u,i} - \lambda b_i)
```

3. Update latent factors:

```math
 \mathbf{p}_u \leftarrow \mathbf{p}_u + \eta (e_{u,i} \mathbf{q}_i - \lambda \mathbf{p}_u)
```

```math
 \mathbf{q}_i \leftarrow \mathbf{q}_i + \eta (e_{u,i} \mathbf{p}_u - \lambda \mathbf{q}_i)
```

Where:
- $` \eta `$ is the learning rate.

### 7. Evaluation

The performance of the model is evaluated using Root Mean Square Error (RMSE):

```math
\text{RMSE} = \sqrt{\frac{1}{|\mathcal{K}|} \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \hat{r}_{u,i})^2}
```
Lower RMSE values indicate better model performance.

# External Resources

- [recommender_inputs.pkl](https://drive.google.com/file/d/1ieoP27pGHIA6eZ1FHlQUwOEQYL2U2JBR/view?usp=sharing)

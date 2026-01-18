### Polyhedron Gravitation

#### GP, GV, GGT expressions

The *gravitational potential (GP)* is  
$$
V = \frac{1}{2} G \rho \sum_{e \in \text{edges}} \mathbf{r}_e \cdot \mathbf{E}_e \cdot \mathbf{r}_e \, L_e - \frac{1}{2} G \rho \sum_{f \in \text{faces}} \mathbf{r}_f \cdot \mathbf{F}_f \cdot \mathbf{r}_f \, \omega_f \tag{1}
$$

The *gravitational vector (GV)* is  
$$
\nabla V = -G \rho \sum_{e \in \text{edges}} \mathbf{E}_e \cdot \mathbf{r}_e \, L_e + G \rho \sum_{f \in \text{faces}} \mathbf{F}_f \cdot \mathbf{r}_f \, \omega_f \tag{2}
$$

The *gravity gradient tensor (GGT)* is  
$$
\nabla \nabla V = G \rho \sum_{e \in \text{edges}} \mathbf{E}_e \, L_e - G \rho \sum_{f \in \text{faces}} \mathbf{F}_f \, \omega_f \tag{3}
$$

where $G$ and $\rho$ represent the gravitational constant and the polyhedron’s constant density. Suffixes $ e $ and $ f $ indicate edge and face, respectively.

##### *Face and edge dyads*: 
$$ 
\mathbf{F}_f = \hat{\mathbf{n}}_{ABC} \hat{\mathbf{n}}_{ABC}, \tag{4}
$$
where $ \hat{\mathbf{n}}_{ABC} $ is the outward-pointing face normal vector for the triangular face $ABC$ (in CCW order). 

$$
\mathbf{E}_{e} = \hat{\mathbf{n}}_{ABC} \hat{\mathbf{n}}_{AB} + \hat{\mathbf{n}}_{BAD} \hat{\mathbf{n}}_{BA}, \tag{5}
$$
where $ABC$ and $BAD$ are the two triangular faces the edge $AB$ belongs to (in CCW order), with $\hat{\mathbf{n}}_{ABC}$ and $\hat{\mathbf{n}}_{BAD}$ the face normal vectors, $\hat{\mathbf{n}}_{AB}$ and $\hat{\mathbf{n}}_{BA}$ the outward-pointing edge normal vectors perpendicular to both the face normal and the edge, i.e., $\hat{\mathbf{n}}_{AB} = \hat{\mathbf{e}}_{AB} \times \hat{\mathbf{n}}_{ABC}$, $\hat{\mathbf{n}}_{BA} = \hat{\mathbf{e}}_{BA} \times \hat{\mathbf{n}}_{BAD}$, with $\hat{\mathbf{e}}_{AB}$ the unit vector along edge $AB$, and $\hat{\mathbf{e}}_{BA}$ the opposite.

$$
\begin{split}
\mathbf{r}_{PA} \cdot \hat{\mathbf{n}}_{AB} 
& = (\mathbf{r}_{A}-\mathbf{r}_{P}) \cdot (\hat{\mathbf{e}}_{AB} \times \hat{\mathbf{n}}_{ABC}) \\
& = \mathbf{r}_{A} \cdot (\hat{\mathbf{e}}_{AB} \times \hat{\mathbf{n}}_{ABC})
    - \mathbf{r}_{P} \cdot (\hat{\mathbf{e}}_{AB} \times \hat{\mathbf{n}}_{ABC}) \\
& = \mathbf{r}_{A} \cdot (\hat{\mathbf{e}}_{AB} \times \hat{\mathbf{n}}_{ABC})
    + \hat{\mathbf{e}}_{AB} \cdot (\mathbf{r}_{P} \times \hat{\mathbf{n}}_{ABC})
\end{split}
$$

##### *Per-edge factor* $L_e$ and *per-face factor* $\omega_f$ 

Considering computation point $P$, *per-edge factor* $L_e$ corresponding to an edge $AB$ is:
$$
L_e = \ln \frac{r_{PA} + r_{PB} + e_{AB}}{r_{PA} + r_{PB} - e_{AB}}, \tag{6}
$$
with $r_{PA}$, $r_{PB}$ and $e_{AB}$ the lengths of $PA$, $PB$ and $AB$.

*Per-face factor* $\omega_f$ corresponding to a triangular face $ABC$ is: 
$$
\omega_f = 2 \arctan \frac{\mathbf{r}_{PA} \cdot (\mathbf{r}_{PB} \times \mathbf{r}_{PC})}
{r_{PA} r_{PB} r_{PC} + r_{PA} (\mathbf{r}_{PB} \cdot \mathbf{r}_{PC}) + r_{PB} (\mathbf{r}_{PC} \cdot \mathbf{r}_{PA}) + r_{PC} (\mathbf{r}_{PA} \cdot \mathbf{r}_{PB})} \tag{7}
$$
which involves inner, cross and mixed vector products.

#### Decoupling computation and polyhedron coordinates

Terms *involving both polyhedron and computation coordinates* needs to be decoupled to simplify vectorized coding.

*Polyhedron geometry*:

- **Vertex list**: $\big[x_{i}, y_{i}, z_{i}\big]_{i=1,\cdots,N_v}$ storing in a $\big[ Q \big]_{N_v \times 3}$ matrix; 
- **Face list**: $\big[i_{A}^{(j)}, i_{B}^{(j)}, i_{C}^{(j)}\big]_{j=1,\cdots,N_f}$ storing in a $\big[I_f\big]_{N_f \times 3}$ matrix, with each row containing $3$ indices to the vertex list corresponding to the triangle $ABC$; 
- **Edge list**: $\big[i_{A}^{(k)}, i_{B}^{(k)}, i_{AB}^{(k)}, i_{BA}^{(k)} \big]_{k=1,\cdots,N_e}$ storing in a $\big[I_e\big]_{N_e \times 4}$ matrix, with each row containing $2$ indices (for edge $AB$) to the vertex list and $2$ indices (for faces $ABC$ and $BAD$) to the face list; 

*Matrix*:

- Computation point list: $\big[x_{m}, y_{m}, z_{m}\big]_{m=1,\cdots,M}$ storing in a $\big[P \big]_{M \times 3}$ matrix; 
- Vertex list points storing in a $\big[Q \big]_{N_v \times 3}$ matrix;
- For $\omega_f$: matrix $\big[A \big]_{N_f \times 3}$, $\big[B \big]_{N_f \times 3}$, $\big[C \big]_{N_f \times 3}$ containing coordinates for all faces $ABC$ (by indexing into $Q$ using $I_f$); 
- For $L_e$: matrix $\big[A \big]_{N_e \times 3}$, $\big[B \big]_{N_e \times 3}$ containing coordinates for all edges $AB$ (by indexing into $Q$ using $I_e$); 

*Vectorized coding*

We have $M$ computation points storing in $\big[P \big]_{M \times 3}$, and $N$ polyhedron vertices storing in the matrix $\big[A \big]_{N \times 3}$, $N=N_f$ when evaluating $\omega_f$, and $N=N_e$ when evaluating $L_e$. 

##### *Pairwise distance*: 

We can compute $M \times N$ pariwise distance $r_{PA}$ ($r_{PB}$ and $r_{PC}$ analogously) as:

$$
r_{PA} = |\mathbf{r}_A-\mathbf{r}_P|=\sqrt{|\mathbf{r}_P|^2 + |\mathbf{r}_A|^2 -2 \mathbf{r}_P \cdot \mathbf{r}_A}
$$

```python
rP2   = np.sum(P**2, axis=1);                         # (M,)
rA2   = np.sum(A**2, axis=1);                         # (N,)
rP_rA = P @ A.T;                                      # (M,3) @ (3,N) --> (M,N)
rPA   = np.sqrt(rP2[:,np.newaxis] + rA2 - 2 * rP_rA); # Broadcasting (M,1), (N,), (M,N) --> (M,N)
```

##### *Inner product*: 

We can compute $M \times N$ pariwise inner product $\mathbf{r}_{PA} \cdot \mathbf{r}_{PB}$ ($\mathbf{r}_{PB} \cdot \mathbf{r}_{PC}$ and $\mathbf{r}_{PC} \cdot \mathbf{r}_{PA}$ analogously) as:

$$
\begin{aligned}
\mathbf{r}_{PA} \cdot \mathbf{r}_{PB} 
& = (\mathbf{r}_A-\mathbf{r}_P) \cdot (\mathbf{r}_B-\mathbf{r}_P) \\
& = |\mathbf{r}_P|^2 + \mathbf{r}_{A} \cdot \mathbf{r}_{B} 
    - \mathbf{r}_P \cdot (\mathbf{r}_A + \mathbf{r}_B)
\end{aligned}
$$

```python
rP2     = np.sum(P**2, axis=1);                # (M,)
rA_rB   = np.sum(A*B, axis=1);                 # (N,)
rP_rArB = P @ (A+B).T;                         # (M,3) @ (3,N) --> (M,N)
rPA_rPB = rP2[:,np.newaxis] + rA_rB - rP_rArB; # Broadcasting (M,1), (N,), (M,N) --> (M,N)
```

##### *Mixed product*: 

$$
\begin{aligned}
\mathbf{r}_{PA} \cdot (\mathbf{r}_{PB} \times \mathbf{r}_{PC})
& = \mathbf{r}_{A} \cdot (\mathbf{r}_{B} \times \mathbf{r}_{C}) 
    - \mathbf{r}_{P} \cdot \big( \mathbf{r}_{A} \times \mathbf{r}_{B} + \mathbf{r}_{B} \times \mathbf{r}_{C} + \mathbf{r}_{C} \times \mathbf{r}_{A}\big)
\end{aligned}
$$

> **Note:** Relation with $\mathbf{r}_f \cdot \hat{\mathbf{n}}_f$

Let
$$
\begin{aligned}
\mathbf{n}_{ABC} & = (\mathbf{r}_B-\mathbf{r}_A) \times (\mathbf{r}_C-\mathbf{r}_B) \\
                 & = \mathbf{r}_{A} \times \mathbf{r}_{B} + \mathbf{r}_{B} \times \mathbf{r}_{C} +  \mathbf{r}_{C} \times \mathbf{r}_{A} = |2S_{\Delta ABC}| \; \hat{\mathbf{n}}_{ABC}
\end{aligned}
$$
be the unnormalized normal vector of $ABC$, with its length equal to twice the area of $ABC$, i.e., $|2S_{\Delta ABC}|$, then we have: 
$$
\mathbf{r}_{PA} \cdot \hat{\mathbf{n}}_{ABC} 
= \frac{(\mathbf{r}_A-\mathbf{r}_P) \cdot \mathbf{n}_{ABC}}{|2S_{\Delta ABC}|}
= \frac{\mathbf{r}_{PA} \cdot (\mathbf{r}_{PB} \times \mathbf{r}_{PC})}{|2S_{\Delta ABC}|}
$$

#### Further accelerate:

**By indexing into $\big[ PQ^T \big]_{M\times N_v}$, better!!!**

```python

```

```python

```

--------
### References
- Werner, R.A., Scheeres, D.J. Exterior gravitation of a polyhedron derived and compared with harmonic and mascon gravitation representations of asteroid 4769 Castalia. Celestial Mech Dyn Astr 65, 313–344 (1996). https://doi.org/10.1007/BF00053511
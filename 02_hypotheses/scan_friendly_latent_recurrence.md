# Scan-Friendly Sequence-Axis Latent Recurrence

Status: **proposed / unvalidated**

This note records a candidate architecture family motivated by the current project conclusion:

> Sequence-axis latent recurrence is difficult to evaluate at scale because exact nonlinear token-by-token recurrence destroys Transformer-style sequence parallelism. The next useful architecture should make the recurrent transition itself composable, so that temporal state can be computed with an exact or near-exact parallel scan.

The central design principle is:

[
\boxed{
\text{scan states}
\;\longrightarrow\;
\textbf{scan composable transition operators}
}
]

Instead of trying to parallelize an arbitrary nonlinear recurrence

[
s_t = F_\theta(h_t,s_{t-1}),
]

we constrain the state-transition family to be closed under composition:

[
s_t = G_{\phi(h_t)}(s_{t-1}),
\qquad
G_b \circ G_a \in \mathcal G.
]

Then every token independently produces a transition operator (g_t), and the full recurrent trajectory can be recovered through an associative prefix scan:

[
g_{1:t}
=
g_t \otimes g_{t-1}\otimes\cdots\otimes g_1.
]

The hypothesis is that most nonlinear modeling capacity can remain in Transformer read/write modules, while the temporal carrier itself stays scan-friendly.

---

## 1. Main candidate: Block-Affine Latent Scan

Let a Transformer hidden state at some middle/high layer be

[
h_t\in\mathbb R^d.
]

A parallel projection generates recurrent transition parameters:

[
(A_t,b_t,g_t)=\phi(h_t).
]

The latent state is split into small blocks:

[
s_t=
[s_t^{(1)},\ldots,s_t^{(G)}],
\qquad
s_t^{(j)}\in\mathbb R^k,
]

with a small block size such as (k\in\{2,4,8\}).

Each block evolves as

[
s_t^{(j)}
=
A_t^{(j)}s_{t-1}^{(j)}
+
b_t^{(j)},
\qquad
A_t^{(j)}\in\mathbb R^{k\times k}.
]

For an affine operator

[
\mathcal O_t=(A_t,b_t),
]

composition is

[
(A_2,b_2)\circ(A_1,b_1)
=
(A_2A_1,\;A_2b_1+b_2).
]

This operator is associative:

[
(\mathcal O_3\circ\mathcal O_2)\circ\mathcal O_1
=
\mathcal O_3\circ(\mathcal O_2\circ\mathcal O_1),
]

so all prefix states can be obtained by parallel scan instead of token-serial forward execution.

### Why block-affine instead of diagonal-only

The diagonal case

[
s_t=a_t\odot s_{t-1}+b_t
]

is the cheapest baseline and should be implemented first, but independent scalar channels may be too restrictive.

Small block matrices permit:

- latent feature mixing inside each subspace;
- rotation / basis change;
- controlled memory decay;
- richer state dynamics without losing composition closure.

The current main architecture hypothesis is therefore:

[
\boxed{
\textbf{4×4 block-affine scan}
+
\textbf{nonlinear Transformer read/write}
}
]

rather than a fully general nonlinear recurrent transition.

---

## 2. Stable parameterization: decay + rotation

A useful restricted parameterization for (2\times2) blocks is

[
A_t
=
\rho_t
\begin{bmatrix}
\cos\theta_t & -\sin\theta_t\\
\sin\theta_t & \cos\theta_t
\end{bmatrix},
\qquad
0<\rho_t\le1.
]

Then

[
s_t=A_ts_{t-1}+b_t.
]

Composition remains cheap:

[
\rho_2R(\theta_2)\rho_1R(\theta_1)
=
(\rho_2\rho_1)R(\theta_1+\theta_2).
]

This gives the recurrent carrier two useful primitives:

- **persistence / forgetting** through \(\rho_t\);
- **latent subspace evolution** through \(\theta_t\).

This variant should be treated as a stability-oriented ablation rather than the only design.

---

## 3. Keep nonlinearity outside the recurrent carrier

A core design choice is to avoid placing an arbitrary MLP/Transformer directly inside the temporal recurrence.

### Write

[
u_t = \mathrm{MLP}_{write}(h_t)
]

[
(A_t,b_t,g_t)=W_{op}u_t
]

### Scan

[
s_t=A_ts_{t-1}+b_t
]

### Read

[
r_t
=
\mathrm{MLP}_{read}
\left(
[h_t,W_ss_t]
\right)
]

### Fuse

[
\tilde h_t
=
h_t
+
\sigma(g_t)\odot r_t.
]

Interpretation:

[
\boxed{
\text{Transformer layers perform nonlinear computation;}
\quad
\text{the recurrent scan transports / reuses latent computation.}
}
]

This differs from RLT, where the recurrent state itself passes through a large nonlinear decoder transition at every token.

The working hypothesis is that constraining the **carrier** may be a better systems/architecture tradeoff than constraining the entire model.

---

## 4. Placement inside a Transformer

The initial version should not replace attention.

It is a side-channel added to a conventional causal Transformer. The intended flow is:

1. token hidden (h_t);
2. normal attention + MLP;
3. middle/high-level feature (z_t);
4. parallel write of transition operator ((A_t,b_t));
5. associative temporal scan producing (s_t);
6. nonlinear read from (s_t);
7. gated residual fusion back into the Transformer stream.

Recommended first placement:

- one scan module in the middle third of the network;
- optionally a second module in the upper third;
- recurrent state width much smaller than model hidden width;
- do not initially place a scan module at every layer.

This keeps the experiment focused on whether a cheap latent temporal side-channel has measurable value.

---

## 5. GPU formulation: chunked fused operator scan

A mathematically associative recurrence is not automatically GPU-efficient. The implementation target should look like modern SSM / linear-attention kernels rather than a Python-level prefix loop.

For sequence length (T) and chunk size (C):

### Step A — parallel parameter generation

Use large GEMMs over all tokens:

[
H\in\mathbb R^{BT\times d}
\rightarrow
\{A_t,b_t,g_t\}_{t=1}^T.
]

### Step B — fused intra-chunk scan

Each chunk computes its local prefix states and a summary operator

[
(\bar A_c,\bar b_c)
]

such that

[
s_{\mathrm{end}}
=
\bar A_c s_{\mathrm{start}}+\bar b_c.
]

### Step C — global scan over chunk summaries

Only

[
T/C
]

small operators participate in the cross-chunk prefix scan.

### Step D — parallel reconstruction inside chunks

Once every chunk start state is known, all chunks reconstruct their internal states independently.

The desired execution shape is therefore:

[
\boxed{
\text{large GEMMs}
+
\text{fused local scan}
+
\text{tiny global scan}
}
]

rather than thousands of small sequential kernels.

Backward should likewise use a dedicated reverse scan / fused kernel instead of relying on a long autograd graph.

---

## 6. Architecture variants to test

### A. ScanLR-Diag

[
s_t=a_t\odot s_{t-1}+b_t
]

Purpose:

- implementation baseline;
- fastest possible scan carrier;
- establish throughput ceiling.

### B. ScanLR-Block

[
s_t^{(j)}
=
A_t^{(j)}s_{t-1}^{(j)}
+
b_t^{(j)}
]

with (k=4) as the primary candidate.

Purpose:

- test whether limited latent subspace mixing provides useful additional capacity;
- preserve exact associative scan.

### C. ScanLR-Rotation

Use stable decay+rotation blocks.

Purpose:

- test whether explicitly structured persistence improves long-horizon dynamics;
- reduce instability of arbitrary block matrices.

### D. ScanLR-Hierarchical

Token-level block-affine scan plus slow chunk-level nonlinear recurrence:

[
m_c
=
F_\theta(m_{c-1},\operatorname{Pool}(h_c)).
]

For example (C=64) or (128).

The serial depth becomes (T/C), not (T).

Purpose:

- preserve cheap token-level recurrence;
- allow occasional genuinely nonlinear state updates;
- test whether long-range latent semantics need nonlinear transition at every token.

### E. Projective / Möbius recurrence — high-risk branch

A genuinely nonlinear but composition-closed scalar/group transition:

[
s_t
=
\frac{a_ts_{t-1}+b_t}
{c_ts_{t-1}+d_t}.
]

Represent each transition with

[
M_t=
\begin{bmatrix}
a_t&b_t\\
c_t&d_t
\end{bmatrix}
]

so that composition reduces to matrix multiplication.

Purpose:

- test whether nonlinear state dynamics can remain exact-scan compatible.

Risks:

- denominator instability;
- difficult parameterization;
- less obvious numerical behavior;
- likely not first-line architecture.

This branch should only be attempted if block-affine recurrence shows positive signal.

---

## 7. What would count as success

The architecture is motivated by the project's current conclusion that compute efficiency is a gating condition.

Therefore a quality gain alone is insufficient.

A useful result should jointly satisfy:

[
\text{quality}
+
\text{recurrent utility}
+
\text{training efficiency}.
]

The primary systems target is:

[
\boxed{
\text{ScanLR training throughput}
\gtrsim
0.8\times
\text{vanilla Transformer}
}
]

at matched model size / context / batch, after kernel optimization.

If the method remains around (0.4\times) or below vanilla throughput, it has not really solved the scaling trap.

The intended evaluation should include three Pareto curves:

[
\text{validation loss}
\quad\text{vs}\quad
\text{training FLOPs}
]

[
\text{validation loss}
\quad\text{vs}\quad
\text{wall-clock}
]

[
\text{recurrent utility / persistence}
\quad\text{vs}\quad
\text{wall-clock}.
]

---

## 8. Mechanistic predictions

If the formulation is doing something meaningfully different from ordinary linear-attention memory, we should observe at least some of the following:

1. Turning off the latent scan path should measurably increase NLL after training.
2. Utility should not be confined to a single token if the carrier learns persistent latent state.
3. The state should add information / computation beyond what attention history trivially reconstructs.
4. Block-affine recurrence should outperform diagonal recurrence at similar systems cost if latent subspace mixing matters.
5. Hierarchical nonlinear updates should improve long-range state utility if nonlinear transition is needed only at coarse temporal resolution.
6. The method should preserve most Transformer training throughput.

A failure on (6) is especially important: it means the proposal does not solve the primary research bottleneck even if it produces an architectural gain.

---

## 9. Main falsifiers

The architecture family should be deprioritized if:

- optimized ScanLR remains substantially slower than vanilla Transformer;
- recurrent-path ablation shows near-zero utility after sufficient training;
- gains disappear under compute-matched comparison;
- block-affine variants do not outperform a diagonal scan or ordinary linear-attention side-channel;
- persistence increases without improving language modeling or task quality;
- attention can reconstruct all useful recurrent information with negligible loss.

---

## 10. Research question

The broad architecture question is no longer:

> Can we add a recurrent edge between adjacent tokens?

It is:

[
\boxed{
\textbf{Can a sequence-axis latent state be represented as a composable operator process}
\\
\textbf{so that recurrence keeps useful temporal semantics without sacrificing Transformer-scale parallelism?}
}
]

The first concrete implementation candidate is:

[
\boxed{
\textbf{4×4 Block-Affine Latent Scan}
+
\textbf{nonlinear Transformer read/write}
+
\textbf{chunked fused prefix scan}
}
]

Everything in this note remains **unvalidated** until both the mechanism and the systems target are measured.

# 🐜 Ant Colony Cookie Transport — Mean-Field Control meets Collective Intelligence

> *45 ants. One cookie. One target. No central coordinator.*
> *Just math, noise, and emergent cooperation.*

---

![ADD ANIMATION HERE COMING SOON](ant_colony_phibe.gif)

What you are watching is not hand-coded behaviour. The ants have **no explicit rules** telling them to surround the cookie, attach to it, or pull in the same direction. All of this emerges from a single optimal policy — learned automatically by solving a **Mean-Field Control problem** over the joint distribution of the entire colony.

If you work in RL, multi-agent systems, or stochastic control, there is something here for you. If you are just here because watching ants carry a cookie is satisfying — also valid.

---

## Why ants?

Collective transport in ant colonies is one of the most striking examples of decentralised coordination in nature. Individual ants have no global map, no communication protocol, and no designated roles — yet colonies reliably transport objects many times their size around obstacles and across uneven terrain. Social insect colonies operate without any central control; their collective behavior arises from local interactions among individuals. 

This has motivated a long line of modelling work, from stochastic foraging models (Prabhakar et al., *PLOS Computational Biology*, 2012) to mean-field PDE approaches for robotic swarms (Zheng et al., *IEEE Transactions on Automatic Control*, 2021). Mean-field partial differential equations can model a swarm and control its mean-field density over a bounded spatial domain, with control laws that act locally on individual robots to guide their global distribution. 

Our work is motivated by the same question — **can a principled mathematical framework reproduce this emergent collective intelligence?** — but attacks it from a different angle: **McKean-Vlasov control**.

---

## The dynamical system

Each ant lives in a **5-dimensional state space**:
```
s = (x, z, y)  ∈  ℝ² × ℝ × ℝ²
```

| Variable | Dimension | Meaning |
|----------|-----------|---------|
| `x` | ℝ² | Ant's spatial position |
| `z` | ℝ | Internal attachment level (continuous, relaxed) |
| `y` | ℝ² | Cookie centre — **shared by the whole population** |

The attachment probability is $r = \Lambda(z) = \sigma(z) \in (0,1)$, the logistic sigmoid. When $\Lambda(z) \approx 1$ the ant is firmly latched onto the cookie; when $\Lambda(z) \approx 0$ it is roaming freely.

Each ant takes an action $a = (u, \eta) \in \mathbb{R}^2 \times [0,1]$: a locomotion vector $u$ (with $\|u\| \le u_{\max}$) and an attachment effort $\eta$.

The key modelling choice: $N$ ants evolve as **exchangeable particles** coupled only through their empirical distribution $\mu_t \approx \frac{1}{N}\sum_i \delta_{s_t^i}$. In the $N \to \infty$ limit this becomes a **McKean-Vlasov SDE** — a single stochastic equation whose coefficients depend on the law of its own solution.

### Drift $b = (b_x,\, b_z,\, b_y)$

**Ant position** $b_x$ (the most interesting piece):

$$b_x(s,\mu,a) = \underbrace{u}_{\text{locomotion}} - \underbrace{\nabla(W_{\text{rep}} * \rho)(x)}_{\text{soft repulsion}} + \underbrace{\beta_{\text{seek}}(1-\Lambda(z))\,\psi(\|x-y\|)\,\frac{y-x}{\|y-x\|+\varepsilon}}_{\text{detached: approach cookie}} - \underbrace{\beta_{\text{hold}}\,\Lambda(z)(x-y)}_{\text{attached: stay near cookie}} + \underbrace{\beta_{\text{pull}}\,\Lambda(z)\,\frac{B-y}{\|B-y\|+\varepsilon}}_{\text{attached: lean toward target}}$$

where $\psi(d) = e^{-d^2/\ell^2}$ is a proximity weight and $B \in \mathbb{R}^2$ is the target.
Five forces act on each ant simultaneously: voluntary locomotion, soft crowd repulsion (via the population marginal $\rho$), cookie attraction (only when detached), elastic attachment spring (only when attached), and a forward lean toward the target (only when attached).

**Attachment dynamics** $b_z$ — an Ornstein-Uhlenbeck relaxation:

$$b_z(s,\mu,a) = \kappa\bigl(c_{\text{sat}}\cdot\eta\cdot\psi(\|x-y\|) - z\bigr)$$

Attachment grows when the ant is close to the cookie **and** applies effort $\eta > 0$. It decays automatically when the ant moves away. The constant $c_{\text{sat}} = 5$ ensures $\Lambda(5) \approx 0.993$ — a fully committed ant is effectively glued.

**Cookie dynamics** $b_y$ — overdamped (high-friction) transport:

$$b_y(\mu) = \frac{1}{\gamma_c}\,\mathcal{F}(\mu), \qquad \mathcal{F}(\mu) = F_0 \int \Lambda(z)\,\psi(\|x-y\|)\,\frac{x-y}{\|x-y\|+\varepsilon}\,\mu(dx,dz,dy)$$

The cookie moves only through the **mean-field force** $\mathcal{F}(\mu)$: the population-averaged traction from all attached ants near the cookie. Notice $b_y$ depends on $\mu$ but **not on any individual state $s$** — this is exactly the McKean-Vlasov structure.

### Diffusion $\sigma$

Only the ant's spatial and attachment degrees of freedom are stochastic:

$$\sigma = \text{diag}(\sigma_x I_2,\; \sigma_z,\; 0_{2\times 2})$$

with $\sigma_x = 0.22$ and $\sigma_z = 0.04 \ll \sigma_x$, keeping attachment smooth while ant motion is noisy. The cookie trajectory is deterministic given $\mu_t$.

### Reward

Each ant maximises an infinite-horizon discounted reward:

$$r(s,\mu,a) = -\frac{c_y}{2}\|y-B\|^2 - \frac{c_u}{2}\|u\|^2 - \frac{c_\eta}{2}\eta^2 - c_{\text{rep}}\,(W_{\text{rep}}*\rho)(x) - \frac{c_d}{2}\|x-y\|^2$$

The five penalties are: get the cookie to $B$, don't waste energy on locomotion, don't waste energy on attachment, avoid crowding, stay close to the cookie. No term explicitly prescribes coordination — it emerges from the mean-field coupling.

### Parameters at a glance

| Parameter | Value | Role |
|-----------|-------|------|
| $\ell = 1.2$ | proximity scale | effective attachment range |
| $\beta_{\text{seek}} = 3.0$ | seek gain | how fast detached ants approach cookie |
| $\beta_{\text{hold}} = 2.5$ | hold spring | elasticity of attachment |
| $\beta_{\text{pull}} = 2.0$ | pull bias | forward lean of attached ants |
| $F_0 = 5.0$, $\gamma_c = 3.0$ | force / drag | cookie transport speed |
| $W_0 = 0.35$, $\sigma_{\text{rep}} = 0.65$ | repulsion | crowd avoidance radius |
| $N = 45$ ants | swarm size | |
| $\beta = 0.2$ | discount | time preference |


## Obstacle avoidance

The rock in the animation is not a hard constraint — it is a smooth Gaussian repulsive potential added to $b_x$ and $b_y$:

$$F_{\text{rock}}(x) = \frac{A_{\text{rock}}}{\sigma_{\text{rock}}^2}(x - x_{\text{rock}})\,\exp\!\left(-\frac{\|x-x_{\text{rock}}\|^2}{2\sigma_{\text{rock}}^2}\right)$$

This keeps everything $C^\infty$ and compatible with the regularity assumptions of the theory, while producing a realistic detour around the obstacle.



## The algorithm:

> **📄 Paper & full algorithm:** *coming soon*


## Code structure



## References

- Prabhakar, B., Dektar, K. N., & Gordon, D. M. (2012). The regulation of ant colony foraging activity without spatial information. *PLOS Computational Biology*, 8(8), e1002670.
- Zheng, T., Han, Q., & Lin, H. (2021). Transporting robotic swarms via mean-field feedback control. *IEEE Transactions on Automatic Control*, 67(8), 4168–4175.
- Carmona, R., & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications*. Springer.

---
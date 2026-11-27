# Log-space corrosion current prediction and physics-clipped Monte Carlo uncertainty propagation

Machine-learning models predict $\log_{10}(i_{\text{corr}})$ rather than linear $i_{\text{corr}}$, and Monte Carlo uncertainty propagation samples $\log_{10}(i_{\text{corr}}) \sim \mathcal{N}(\mu_{\log i}, \sigma_{\log i}^2)$ before exponentiating ($i_{\text{corr}} = 10^x$). Because corrosion current densities span several orders of magnitude ($10^{-2}$ to $10^{3}\text{ }\mu\text{A/cm}^2$), linear-space regression and additive Gaussian noise would yield unphysical negative currents and negative Corrosion Rates during uncertainty propagation.

To prevent non-physical tail explosions while keeping deterministic polarization curves activation-controlled per ADR-0001:
- $E_{\text{corr}}$ is sampled in linear space within thermodynamic bounds; for bimetallic galvanic couples immersed in the same Solution, the two metals share a positive solution-level correlation ($\rho = 0.6$ by default).
- Strictly positive Tafel slopes ($\beta_a, \beta_c$) are sampled from a median-anchored log-normal distribution clipped to $[20, 500]\text{ mV/dec}$ rather than a linear Gaussian that piles up at the lower bound.
- Monte Carlo dissolution current densities ($i_{\text{corr}}$, $i_{\text{galv}}$) are capped at a physical mass-transport limit $i_{\text{lim}} = 10^5\ \mu\text{A/cm}^2$ as an uncertainty-tail guardrail (without altering the deterministic Wagner–Traud equation).
- Bimetallic galvanic Monte Carlo consistently tracks the nominal Active Metal across all draws, allowing cathodic protection ($\text{AF} < 1$) on polarity-flipped draws instead of mixing in Noble Metal dissolution rates.

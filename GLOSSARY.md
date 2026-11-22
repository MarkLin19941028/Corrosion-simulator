# Corrosion Simulator

A physics-informed and machine-learning-assisted electrochemical corrosion simulator for evaluating metal loss and galvanic coupling risk in semiconductor wet cleaning and etching processes.

## Language

### Materials & Solutions

**Metal**:
A conductive metallic element, barrier compound, or alloy film exposed to a liquid chemical environment (core semiconductor metals Cu, Co, W, Ru, Ti, Ta, Al, Mo, plus extended materials such as TiN, Fe, Ni, Cr, Sn, Zn, and alloys).
_Avoid_: Substrate, electrode, target

**Solution**:
A liquid chemical formulation defined by its oxidizer, acid or complexing agent, corrosion inhibitor, concentrations, and pH, independent of operating temperature or exposure duration.
_Avoid_: Recipe, bath, electrolyte condition

**Exposure Condition**:
The combination of a specific Metal, a Solution, and an operating Temperature that uniquely determines the steady-state electrochemical kinetics at the metal-liquid interface.
_Avoid_: Test case, environment, recipe

**Cleaning Recipe**:
A process specification consisting of a Solution, an operating Temperature, and an Immersion Time used to evaluate cumulative material loss and risk during a wet process step.
_Avoid_: Solution, condition, run

### Electrochemical Kinetics

**Corrosion Potential ($E_{\text{corr}}$)**:
The open-circuit mixed potential (in $\text{V vs. SHE}$) at which the anodic dissolution current density equals the cathodic reduction current density on a single isolated Metal under a given Exposure Condition.
_Avoid_: Open circuit potential (OCP), rest potential, standard potential

**Corrosion Current Density ($i_{\text{corr}}$)**:
The magnitude of the anodic dissolution current density (in $\mu\text{A/cm}^2$) at the Corrosion Potential for an isolated Metal.
_Avoid_: Exchange current density, corrosion current

**Tafel Slopes ($\beta_a, \beta_c$)**:
The base-10 logarithmic potential change per decade of current density (in $\text{mV/dec}$, strictly positive) governing the anodic dissolution branch ($\beta_a$) and cathodic reduction branch ($\beta_c$) of the Polarization Curve.
_Avoid_: Transfer coefficient, symmetry factor

**Polarization Curve**:
The relationship between applied or mixed potential $E$ and net current density $i_{\text{net}}(E)$ modeled via the four-parameter activation-controlled Wagner-Traud equation around $(E_{\text{corr}}, i_{\text{corr}}, \beta_a, \beta_c)$.
_Avoid_: I-V curve, voltammogram

**Corrosion Rate ($\text{CR}$)**:
The instantaneous thickness loss rate of a Metal (in $\text{nm/min}$ or $\text{mm/yr}$) derived from its dissolution current density via Faraday's law.
_Avoid_: Etch rate, dissolution speed

**Corrosion Depth**:
The cumulative thickness of Metal lost (in $\text{nm}$) after a given Immersion Time.
_Avoid_: Etch depth, weight loss

### Galvanic Coupling

**Active Metal**:
In a electrically coupled bimetallic pair under the same Solution and Temperature, the Metal with the more negative Corrosion Potential ($E_{\text{corr}}$), which acts as the net anode and undergoes accelerated dissolution.
_Avoid_: Sacrificial metal, base metal, corroding metal

**Noble Metal**:
In a bimetallic pair, the Metal with the more positive Corrosion Potential ($E_{\text{corr}}$), which acts as the net cathode.
_Avoid_: Cathode metal, passive metal

**Mixed Potential ($E_{\text{galv}}$)**:
The steady-state galvanic coupling potential (in $\text{V vs. SHE}$) of two electrically connected Metals at which the area-weighted net current across both surfaces sums to zero.
_Avoid_: Couple potential, corrosion potential

**Galvanic Current Density ($i_{\text{galv}}$)**:
The anodic dissolution current density (in $\mu\text{A/cm}^2$) on the Active Metal surface when polarized to the Mixed Potential.
_Avoid_: Couple current, net current

**Acceleration Factor**:
The dimensionless ratio of the Active Metal's Galvanic Current Density when coupled ($i_{\text{galv}}$) to its isolated Corrosion Current Density ($i_{\text{corr, active}}$).
_Avoid_: Galvanic ratio, amplification factor

### Time-Resolved Kinetics & Risk

**Initial Corrosion Rate ($\text{CR}_0$)**:
The instantaneous Corrosion Rate at $t = 0$ derived directly from Faraday's law using the steady-state $i_{\text{corr}}$ (for an isolated Metal) or $i_{\text{galv}}$ (for an Active Metal in a galvanic couple).
_Avoid_: Average corrosion rate, steady etch rate

**Passivation Limiting Depth ($d_{\max}$)**:
The asymptotic maximum Corrosion Depth (in $\text{nm}$) reached when a self-limiting passive oxide film forms on the Metal surface under the Passivation kinetics model.
_Avoid_: Oxide thickness, saturation depth

**Critical Thickness Threshold ($d_{\text{crit}}$)**:
The maximum tolerable Corrosion Depth (in $\text{nm}$, default $1.0\text{ nm}$) allowed by a semiconductor process step before device performance or structural integrity is compromised.
_Avoid_: Spec limit, failure depth

**Time to Threshold**:
The Immersion Time required for a Metal's Corrosion Depth $d(t)$ to reach the Critical Thickness Threshold ($d_{\text{crit}}$).
_Avoid_: Lifetime, failure time

**Galvanic Driving Force Risk**:
A qualitative screening tier (`Low`, `Medium`, `High`) based solely on the absolute open-circuit Corrosion Potential difference $|\Delta E_{\text{corr}}|$ between two uncoupled Metals in the same Solution and Temperature.
_Avoid_: Galvanic risk, couple risk

**Thickness Loss Risk**:
A time-dependent risk evaluation consisting of a continuous score $R(t) \in [0, 1]$ and a discrete tier (`Low`, `Medium`, `High`, `Critical`, or `N/A` when physical properties required for Faraday conversion are unavailable) determined by the ratio of cumulative Corrosion Depth $d(t)$ to the Critical Thickness Threshold $d_{\text{crit}}$.
_Avoid_: Corrosion risk, damage score

**Applicability Domain**:
The region of Metal and Exposure Condition feature space sufficiently covered by the experimental training database where machine-learning predictions are considered interpolation rather than extrapolation.
_Avoid_: Valid range, confidence interval

**Dissolution Valence ($z$)**:
The number of electrons transferred per dissolved Metal atom in Faraday's law (e.g., $\text{Cu} \to 2$, $\text{Co} \to 2$, $\text{W} \to 6$, $\text{Ru} \to 4$, $\text{Ti} \to 4$, $\text{Ta} \to 5$, $\text{Al} \to 3$, $\text{Mo} \to 6$).
_Avoid_: Charge number, stoichiometry

**Confidence Tier**:
A discrete reliability rating (`High`, `Medium`, `Low`) assigned to an ML-predicted Exposure Condition based on whether its features lie within the training Bounding Box and the local k-NN density of the Applicability Domain.
_Avoid_: Accuracy score, certainty level

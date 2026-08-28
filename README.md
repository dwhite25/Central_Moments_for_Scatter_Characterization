# Statistical Moment Estimation for Sub-Resolution Scatterer Characterization

This repository contains the simulation, estimation, Fisher-information, and analysis code developed for the manuscript:

**"A Moment-Based Framework for Sub-Resolution Radar Scatterer Characterization"**

The project investigates how spatially distributed scattering targets can be characterized when their internal structure lies far below the conventional inverse-bandwidth range-resolution scale.

Rather than attempting to reconstruct an unresolved scattering distribution point by point, the method estimates its **low-order statistical moments** — mean, standard deviation, skewness, and kurtosis — directly from its band-limited return spectrum.

The central idea is that although fine spatial structure becomes increasingly inaccessible as a target becomes deeply sub-resolution, its lowest-order moments continue to produce measurable perturbations in the received signal.

---

## Scientific Motivation

For an interrogating waveform with bandwidth $B$, conventional range resolution is associated with a scale on the order of $B^{-1}$ in round-trip delay, or $\frac{c}{2B}$ in range.

When an entire scattering distribution is substantially smaller than this scale, its individual scattering features cannot generally be resolved as distinct peaks. Direct reconstruction of the underlying distribution therefore becomes increasingly ill-conditioned.

This does not imply that all sub-resolution information has disappeared.

For a normalized scatter distribution $h(t)$, its spectrum can be expanded around zero frequency as

$H(\omega) = \sum_{n=0}^{\infty}\frac{(-i\omega)^n}{n!}\mu_n$,

where $\mu_n$ is the $n$-th raw moment of the distribution.

As the spatial width of the scatterer decreases, successive terms in this expansion become progressively smaller. The received spectrum is therefore governed primarily by the first few moments of the scatter distribution.

This motivates a different estimation objective:

> *Instead of reconstructing every sub-resolution scattering feature, estimate the statistical quantities that remain identifiable through the finite measurement bandwidth.*

---

## Estimation Method

For a known interrogating spectrum $G(\omega)$, scatter spectrum $H(\omega)$, and measured return spectrum $R(\omega)$,

$R(\omega)=G(\omega)H(\omega)$.

The scatter spectrum is estimated at the interrogating frequencies through

$H(\omega_k) = \frac{R(\omega_k)}{G(\omega_k)}$.

A truncated Taylor expansion is then fit to the sampled complex spectrum using least squares,

$H(\omega) \approx 1+\sum_{n=1}^{N}\frac{(-i\omega)^n}{n!}\mu_n$.

The fitted coefficients provide estimates of the raw moments, which are converted to the corresponding statistical quantities:

- mean
- standard deviation
- skewness
- kurtosis

Both fourth- and sixth-order spectral models are examined. The sixth-order model includes higher-order terms in the regression while retaining the first four moments as the primary quantities of interest.

The simulations additionally evaluate estimator performance under additive Gaussian noise and compare the resulting RMSE with Cramer-Rao lower bounds derived from the same spectral model.

---

## Simulation Study

The primary numerical study uses a large database of distributed scattering targets constructed from two or three weighted point scatterers.

The point locations and amplitudes are systematically varied to produce approximately **688,000 candidate scatter distributions** spanning a wide range of means, widths, skewnesses, and kurtoses.

A second database is constructed using two-component, equal-variance **Gaussian mixture models (GMMs)**. Each GMM is numerically fitted to reproduce the low-order moment statistics of a corresponding discrete scatterer.

This provides two physically different classes of scatter distributions with similar statistical descriptions and allows the moment-estimation framework to be tested beyond a specific point-scatterer representation.

Simulated measurements are generated at multiple signal-to-noise ratios using a six-frequency interrogating waveform. Fourth- and sixth-order spectral fits are then used to estimate the underlying scatterer moments.

---

## Repository Workflow

The numerical study is organized into four sequential notebooks:

#### [`01_create_scatterer_database.ipynb`](01_create_scatterer_database.ipynb)

Constructs the source scatterer databases.

- generates two- and three-point discrete scatter distributions
- calculates their statistical moments
- fits corresponding two-component GMM distributions
- validates the GMM moment matches

Outputs:

`delta_database.csv`
`gmm_database.csv`

#### [`02_generate_measurements.ipynb`](02_generate_measurements.ipynb)

Generates simulated measurements and estimates scatterer moments.

- constructs the interrogating waveform
- convolves each waveform with the scatter distributions
- adds noise at the selected SNR
- extracts the scatter spectrum from the simulated return
- fits the spectral Taylor expansion
- estimates mean, standard deviation, skewness, and kurtosis

Outputs are written to:

`data/returns/`

#### [`03_fisher_information.ipynb`](03_fisher_information.ipynb)

Evaluates the Fisher information and Cramer-Rao lower bounds associated with the spectral moment model.

The notebook computes CRLBs both with the return amplitude treated as known and with amplitude included as an additional nuisance parameter.

The Fisher-information condition number is also recorded to identify regimes in which the inverse problem becomes numerically ill-conditioned.

Outputs are written to:

`data/crlb/`

#### [`04_results_and_figures.ipynb`](04_results_and_figures.ipynb)

Analyzes the generated databases and reproduces the primary numerical results.

The analysis includes:

- estimator failure rates
- moment-estimation RMSE versus scatterer width
- SNR dependence
- fourth- versus sixth-order spectral fits
- discrete versus GMM scatterers
- RMSE versus Cramer-Rao bounds
- Fisher-information conditioning

---

## Source Code

Reusable simulation and estimation tools are contained in `src/`.

[`base_waves.py`](src/base_waves.py) -- Waveform construction, Fourier-domain signal representation, convolution with scatter distributions, normalization, and noise generation.

[`databases.py`](src/databases.py) -- Configuration and chunked generation of the large simulated measurement databases. Long calculations are written incrementally so interrupted runs can resume from the existing output file.

[`gmm.py`](src/gmm.py) -- Analytic moment calculations and numerical fitting of two-component Gaussian mixture models to target statistical moments.

[`moments.py`](src/moments.py) -- Numerical calculation of mean, variance, standard deviation, skewness, kurtosis, and higher standardized moments of sampled distributions.

[`scatterers.py`](src/scatterers.py) -- Representations of discrete point-scatterer and Gaussian-mixture scattering distributions.

[`spectral_estimation.py`](src/spectral_estimation.py) -- Core signal-processing and moment-estimation routines, including:

- Fourier coefficient extraction
- scatter-spectrum recovery
- complex least-squares fitting
- common-phase estimation
- truncated spectral Taylor models
- conversion between raw and standardized moments
- optional cumulant-domain estimation
- diagnostic and correlation utilities

---

## Repository Structure

```text
Central_Moments_for_Scatter_Characterization/
├── README.md
├── LICENSE
├── requirements.txt
├── 01_create_scatterer_database.ipynb
├── 02_generate_measurements.ipynb
├── 03_fisher_information.ipynb
├── 04_results_and_figures.ipynb
└── src/
    ├── base_waves.py
    ├── databases.py
    ├── gmm.py
    ├── moments.py
    ├── scatterers.py
    └── spectral_estimation.py
```

Generated databases are intentionally excluded from version control.

Running the notebooks creates:

```text
delta_database.csv
gmm_database.csv

data/
├── returns/
└── crlb/
```

---

## Installation

Clone the repository and install the required Python packages:

```text
git clone https://github.com/dwhite25/Central_Moments_for_Scatter_Characterization.git
cd Central_Moments_for_Scatter_Characterization
pip install -r requirements.txt
```

The primary dependencies are:

- NumPy
- pandas
- SciPy
- Matplotlib
- Jupyter

Run the notebooks from the repository root so that the modules in `src/` and the generated databases can be located correctly.

---

## Reproducing the Numerical Study

Run each notebook sequentially.

The complete simulation is computationally expensive. Database generation is performed incrementally, and the long-running routines in notebooks 02 and 03 can resume from previously generated output files.

For initial testing, the notebooks also contain small validation calculations that can be run without generating the complete databases.

---

## Numerical Conditioning

Evaluation of the Cramer-Rao lower bound requires inversion of the Fisher information matrix.

For sufficiently narrow scatterers, higher-order moment directions become weakly distinguishable over the finite measurement bandwidth and the Fisher matrix becomes increasingly ill-conditioned.

The code therefore records the Fisher-information condition number and evaluates the inverse using a Moore–Penrose pseudoinverse. CRLB behavior in strongly ill-conditioned regions should be interpreted as a numerical-conditioning effect rather than a physical resolution threshold.

---

## Technologies and Methods

#### Signal Processing
- Fourier analysis
- band-limited convolution
- complex spectral estimation
- Taylor-series signal models
- least-squares parameter estimation
- additive-noise simulation
#### Statistical Estimation
- raw and standardized statistical moments
- Fisher information
- Cramer-Rao lower bounds
- nuisance-parameter analysis
- numerical conditioning
#### Scientific Computing
- Python
- NumPy
- SciPy
- pandas
- Matplotlib
- large-scale simulation databases
- chunked and resumable numerical workflows

---

## Paper

This repository accompanies:

*D. D. White et al., **"A Moment-Based Framework for Sub-Resolution Radar Scatterer Characterization."***

[Manuscript / preprint link to be added]

If you use this repository or build on this work, please cite the associated paper.

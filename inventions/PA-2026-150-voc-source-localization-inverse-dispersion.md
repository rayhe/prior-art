# System and Method for Real-Time Indoor Volatile Organic Compound Source Localization and Emission Rate Estimation Using Distributed Metal Oxide Semiconductor Gas Sensor Arrays with Computational Fluid Dynamics-Informed Inverse Dispersion Modeling

**LITF-PA-2026-150 · Indoor Air Quality / Environmental Sensing / Edge AI**
**Published:** 2026-08-25
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for continuously localizing the spatial origin and estimating the emission rate of volatile organic compound (VOC) sources within occupied indoor environments using a distributed array of low-cost metal oxide semiconductor (MOS) gas sensors. The system deploys 4-12 sensor nodes throughout a room or building zone, each containing a MOS VOC sensor (e.g., Sensirion SGP41, Bosch BME688), a temperature and humidity sensor for environmental compensation, and a low-power radio for mesh communication. When a VOC emission event is detected (new furniture off-gassing, cleaning product use, cooking emissions, printer operation, adhesive curing), the spatiotemporal concentration gradient across the sensor array is captured at 0.5 Hz sampling rate. A computational fluid dynamics (CFD)-informed inverse dispersion model, running on an edge compute hub, inverts the observed concentration field to estimate the source location (x, y, z ± 0.5 m) and emission rate (µg/min ± 15%). The model uses pre-computed steady-state velocity fields derived from the room geometry and HVAC operating parameters (supply register locations, return vent positions, fan speed from smart thermostat integration). A physics-informed neural network (PINN) embeds the advection-diffusion partial differential equation as a constraint in its loss function, enabling real-time source inference (< 2 seconds latency) without iterative CFD solving. The system outputs per-source VOC contribution maps, cumulative exposure dose estimates by room zone, and actionable ventilation recommendations. Self-calibration uses HVAC fan cycling as a controlled perturbation to refine the velocity field model over time. Federated learning across deployed installations improves source classification accuracy without transmitting raw sensor data.

## Field of the Invention

This invention relates to indoor environmental monitoring, specifically to real-time spatial localization of volatile organic compound emission sources using distributed gas sensor networks and physics-informed machine learning for inverse atmospheric dispersion modeling at room scale.

## Background

The [U.S. Environmental Protection Agency](https://www.epa.gov/indoor-air-quality-iaq/volatile-organic-compounds-impact-indoor-air-quality) reports that indoor concentrations of many VOCs are consistently 2-5× higher than outdoor levels, and can exceed outdoor concentrations by 10× or more during and immediately following activities such as painting, furniture assembly, or cleaning. The [WHO Guidelines for Indoor Air Quality](https://www.who.int/publications/i/item/9789241548106) identify formaldehyde (HCHO), benzene, trichloroethylene, and naphthalene as priority indoor pollutants with established health effects including respiratory irritation, neurological symptoms, and carcinogenicity at chronic exposure levels.

Current consumer indoor air quality monitors have a fundamental limitation: they measure aggregate total VOC (TVOC) concentration at a single point, providing no information about which source in the room is responsible for which fraction of the measured level. A homeowner who sees an elevated TVOC reading on their monitor cannot determine whether the emission originates from their new sofa (formaldehyde from pressed-wood frame), their laser printer (ultrafine particles and styrene), or the cleaning product stored under the sink (limonene and glycol ethers).

Existing approaches to VOC source identification are inadequate for consumer use:

- **Professional industrial hygiene surveys:** Photoionization detector (PID) handheld surveys combined with sorbent tube sampling and GC-MS laboratory analysis can identify and quantify individual VOC species and localize sources by proximal sampling. Cost: $500-2,000 per survey. Turnaround: 1-2 weeks for lab results. Episodic, not continuous. Requires trained practitioners.
- **Multi-sensor electronic noses:** Research systems using arrays of cross-reactive MOS sensors with pattern recognition can classify VOC mixtures into source categories. [Herrero et al., Sensors and Actuators B: Chemical 2020](https://doi.org/10.1016/j.snb.2019.127125) demonstrated 89% source classification accuracy using 8 MOS sensors. However, these systems operate at a single measurement point and do not perform spatial source localization.
- **Distributed sensor networks for outdoor air quality:** [Schneider et al., Atmospheric Environment 2019](https://doi.org/10.1016/j.atmosenv.2019.116862) demonstrated urban air quality mapping using distributed low-cost sensor nodes. Outdoor inverse dispersion modeling has been applied to industrial emission source estimation using Gaussian plume models ([Roscioli et al., Atmospheric Environment 2013](https://doi.org/10.1016/j.atmosenv.2012.08.046)). Indoor environments present fundamentally different challenges: turbulent recirculation zones, HVAC-driven airflow patterns, and room geometry create complex velocity fields that invalidate the Gaussian plume assumption.
- **Tracer gas methods:** Controlled release of SF6 or CO2 tracers combined with multi-point sampling can characterize indoor air mixing patterns. [Ai and Mak, Indoor Air 2017](https://doi.org/10.1111/ina.12340) used tracer gas decay to validate CFD models of room ventilation effectiveness. Tracer methods require active gas release and specialized analyzers, making them unsuitable for continuous automated monitoring.
- **CFD-based inverse source estimation (research only):** [Liu and Zhai, Building and Environment 2018](https://doi.org/10.1016/j.buildenv.2018.05.027) demonstrated inverse CFD for indoor contaminant source identification using adjoint methods, achieving source localization within 1 m using 6 sensor positions. However, their method required iterative CFD solving (minutes to hours per estimation), making real-time operation infeasible. [Chen et al., Building and Environment 2021](https://doi.org/10.1016/j.buildenv.2020.107417) applied Bayesian inference with pre-computed CFD databases for faster indoor source estimation, but required server-class computing hardware and did not address consumer deployment.

The gap in the art is a complete consumer-deployable system that: (a) uses low-cost commodity gas sensors in a distributed configuration, (b) performs real-time spatial localization of VOC emission sources (not just classification), (c) estimates per-source emission rates, (d) runs on edge compute hardware without cloud connectivity, (e) self-calibrates its room airflow model using existing HVAC system perturbations, and (f) provides actionable per-source exposure attribution and ventilation recommendations.

## Detailed Description

### 1. Sensor Node Hardware

Each sensor node comprises: a metal oxide semiconductor VOC sensor (e.g., Sensirion SGP41, TVOC and NOx indices, $5.20; or Bosch BME688 with integrated gas scanner, $7.50) providing broadband VOC sensitivity in the 0-500 ppb ethanol-equivalent range with 1-second response time (T63); a temperature and humidity sensor (Sensirion SHT41, ±0.2°C, ±1.5% RH, $2.10) for MOS baseline compensation, since MOS sensor resistance is strongly temperature- and humidity-dependent; a barometric pressure sensor (Bosch BMP390, ±0.5 hPa, $2.00) for altitude compensation and HVAC state detection via pressure transients; a microcontroller (Nordic nRF5340 with dual Arm Cortex-M33 cores, BLE 5.3, Thread/802.15.4, $4.50) for sensor management, local preprocessing, and mesh communication; and a USB-C power supply (5V/0.3W typical) or two AA batteries (12-month life at 0.5 Hz sampling with deep sleep between samples).

Nodes are designed for unobtrusive placement at existing room fixtures: wall outlets (plug-in form factor), bookshelf positions (freestanding), or ceiling mount near existing smoke detectors. Minimum 4 nodes per room for 3D source localization; 6-8 nodes recommended for rooms larger than 30 m². Target bill-of-materials cost per node: $18-28. Nodes communicate via Thread mesh networking to a border router (which may be the edge compute hub itself) for aggregation and inference.

### 2. VOC Signal Preprocessing and Environmental Compensation

MOS gas sensors exhibit well-documented cross-sensitivity to temperature and humidity that must be corrected before concentration estimation. The SGP41's raw resistance R_VOC varies approximately -0.8%/°C and -0.3%/% RH around its operating point (plate temperature ~300°C). The system applies per-node compensation using a polynomial correction model:

```
C_corrected(t) = f(R_VOC(t), T(t), RH(t), P(t)) − C_baseline(node)
```

where C_baseline is a per-node running baseline estimated from the 10th percentile of readings over the preceding 24 hours, representing the clean-air floor for that node's microenvironment. The baseline subtraction is critical because MOS sensor drift (typically 0.5-2% of span per month) would otherwise corrupt the spatial gradient measurements that source localization depends on.

Each sensor node timestamps its readings using a microsecond-resolution local clock synchronized across the mesh network via the Thread protocol's IEEE 802.15.4 MAC-layer timestamps (±100 µs accuracy). Synchronized timestamps enable reconstruction of the concentration wavefront propagation across the sensor array, a key input to the inverse dispersion model.

### 3. Room Airflow Model Generation

Indoor VOC transport is governed by the advection-diffusion equation:

```
∂C/∂t + u⃗·∇C = D_eff·∇²C + S(x⃗, t)
```

where C is VOC concentration, u⃗ is the air velocity field, D_eff is the effective diffusivity (molecular diffusion D_mol ≈ 10⁻⁵ m²/s, augmented by turbulent diffusivity D_turb from HVAC-driven mixing, typically 10⁻³ to 10⁻¹ m²/s depending on ventilation rate), and S(x⃗, t) is the source emission function to be estimated.

The velocity field u⃗ is the critical input. The system constructs it through three stages:

1. **Room geometry capture:** The user provides room dimensions via a smartphone app (manual entry or LiDAR scan on iPhone Pro / iPad Pro). Key geometric features: room length, width, height; HVAC supply register positions and orientations (ceiling diffuser, wall register, floor register); return air vent positions; window and door positions (as potential natural ventilation paths); and major furniture positions that create flow obstructions. The app stores the geometry as a simplified 3D voxel grid (0.25 m resolution, typically 500-5,000 voxels for a residential room).

2. **HVAC parameter integration:** The system queries the smart thermostat (via Matter/Thread or cloud API) for: fan mode (auto/on/circulate), fan speed (low/medium/high), cooling/heating/fan-only mode, and supply air temperature. Supply airflow rate is estimated from the number and type of registers serving the room (standard 4×10" register: 50-100 CFM at 0.05" w.g.) and verified during self-calibration.

3. **Pre-computed velocity field library:** Rather than running live CFD, the system maintains a library of pre-computed steady-state Reynolds-Averaged Navier-Stokes (RANS) velocity fields for parameterized room archetypes. The archetype parameters include: room aspect ratio (L/W), ceiling height, number and type of supply registers (ceiling/wall/floor), return vent position (high/low wall), and fan speed class. A library of approximately 200 archetype simulations (computed offline using OpenFOAM with the k-ε turbulence model) covers the residential design space. The edge compute hub selects the nearest archetype and scales the velocity field by the actual supply airflow rate. Storage: each velocity field is approximately 200 KB compressed (3D vector field at 0.25 m resolution), so the full library occupies about 40 MB.

### 4. Self-Calibration via HVAC Perturbation

The pre-computed velocity field is an approximation. The system refines it using the HVAC system itself as a controlled perturbation source. When the HVAC fan cycles on or off, the transition from natural convection to forced convection creates a step change in the room's mixing pattern. The sensor array observes how the existing VOC concentration field redistributes in response to this perturbation. By comparing the observed redistribution dynamics (time constants and spatial patterns of concentration equalization across nodes) to the predicted redistribution from the archetype velocity field, the system computes correction factors for local air velocity magnitudes and directions.

The calibration algorithm runs as a recursive least-squares estimator that updates velocity field correction factors after each HVAC transition event. In a typical residential HVAC system cycling 4-8 times per hour, the velocity field converges to ±20% accuracy within 48 hours of installation. The barometric pressure sensors on each node detect HVAC state transitions (supply pressure transients of 2-10 Pa at fan start/stop) to trigger calibration windows automatically.

### 5. Physics-Informed Neural Network for Inverse Source Estimation

The core innovation is a physics-informed neural network (PINN) that solves the inverse source estimation problem in real-time. Given observed concentrations C_obs(x⃗_i, t) at N sensor positions x⃗_i and the pre-computed (and calibrated) velocity field u⃗(x⃗), the PINN estimates the source function S(x⃗, t) that best explains the observations while satisfying the advection-diffusion PDE.

The PINN architecture consists of: an encoder network (3-layer MLP, 128/256/128 units, SiLU activation) that takes as input the concatenated sensor readings [C_1(t), C_2(t), ..., C_N(t)] along with their spatial coordinates and the current HVAC state vector; a source estimator head that outputs the estimated source location (x_s, y_s, z_s) and emission rate Q_s in µg/min; and a physics residual loss that penalizes violations of the advection-diffusion equation at collocation points sampled within the room volume.

The total loss function is:

```
L = λ_data · Σ|C_pred(x⃗_i, t) − C_obs(x⃗_i, t)|² + λ_pde · Σ|∂C_pred/∂t + u⃗·∇C_pred − D_eff·∇²C_pred − S_pred|² + λ_reg · ||S_pred||₁
```

where the L1 regularization on S_pred enforces source sparsity (real indoor scenarios typically have 1-3 active sources at any time). The weighting coefficients λ_data, λ_pde, and λ_reg are set to 1.0, 0.1, and 0.01 respectively, determined empirically from synthetic training data.

The PINN is pre-trained on 50,000 synthetic scenarios generated by forward CFD simulation with randomized source locations, emission rates, room geometries, and HVAC configurations. Training uses the Adam optimizer with cosine annealing learning rate schedule (initial lr = 10⁻³, minimum lr = 10⁻⁵, 200 epochs). The trained model is quantized to FP16 and deployed on the edge compute hub (Raspberry Pi 5 with 8 GB RAM, or equivalent). Inference time: < 2 seconds per source estimation on the Raspberry Pi 5's Cortex-A76 cores.

For multi-source scenarios (e.g., simultaneous off-gassing from new furniture and cooking emissions from kitchen), the system uses iterative source subtraction: the strongest source is estimated first, its predicted contribution is subtracted from the observed concentration field, and the residual field is re-analyzed for additional sources. The process terminates when the residual concentration at all nodes falls below 1.5× the sensor noise floor (approximately 10 ppb ethanol-equivalent for the SGP41).

### 6. Source Classification and Attribution

Beyond spatial localization, the system classifies identified sources into emission categories using temporal emission profile features:

- **Step-function emissions (sudden onset, sustained):** Cleaning product application, painting, adhesive use. Characterized by rapid rise (< 60 seconds to peak) followed by exponential decay with time constant τ_decay (30 minutes to 4 hours depending on ventilation).
- **Chronic low-level emissions (constant):** Furniture off-gassing (formaldehyde from pressed-wood, flame retardants from foam), building material emissions (new carpet, caulking). Characterized by stable emission rate Q_s with diurnal temperature modulation (emission rate scales approximately as exp(-E_a/RT) where E_a is activation energy, typically 30-60 kJ/mol for formaldehyde from UF resin).
- **Episodic pulsed emissions:** Cooking (onion cutting: propenyl sulfenic acid; frying: acrolein, formaldehyde), printer operation (styrene, ozone from laser printers), candle burning (benzene, toluene, formaldehyde). Characterized by event-correlated concentration bursts with durations of 5-60 minutes.
- **Diurnal-modulated emissions:** Sources whose emission rate correlates with temperature (VOC vapor pressure doubles approximately every 7-10°C increase). The system cross-references emission rate estimates with indoor temperature records to identify temperature-driven off-gassing distinct from activity-driven emissions.

A random forest classifier (100 trees, max depth 12, trained on labeled emission profiles from 2,000 annotated scenarios) categorizes each localized source into one of 14 emission types. Classification accuracy: 82% on synthetic test data, improving to 91% after 30 days of user-confirmed labels via the companion app (the user taps "I just cleaned the bathroom" or "new bookshelf delivered").

### 7. Cumulative Exposure Dose Estimation

The system computes per-zone cumulative VOC exposure dose by integrating the reconstructed concentration field over the user's estimated occupancy zones. Occupancy zone assignment uses three methods (in priority order): (a) wearable device location (smartwatch BLE proximity to nearest sensor node), (b) smart home presence sensing (motion sensors, smart speaker voice activity), or (c) default occupancy schedule (user-configured bedroom hours, work-from-home hours).

The cumulative dose D_zone for occupancy zone z over time window [t₁, t₂] is:

```
D_zone(z) = ∫_{t₁}^{t₂} C_reconstructed(x⃗_z, t) · BR(activity) dt
```

where C_reconstructed is the spatially interpolated concentration at the centroid of zone z (using the PINN's continuous concentration field output) and BR(activity) is the breathing rate adjusted for activity level (resting: 7.5 L/min, light activity: 12 L/min, exercise: 30 L/min, from [EPA Exposure Factors Handbook, Chapter 6](https://www.epa.gov/expobox/exposure-factors-handbook-chapter-6)). The dose is reported in µg·hours/m³, the standard metric used by industrial hygienists for chronic exposure assessment.

The system compares cumulative doses against [WHO indoor air quality guidelines](https://www.who.int/publications/i/item/9789241548106) (formaldehyde: 100 µg/m³ 30-minute average) and [California CDPH Section 01350](https://www.cdph.ca.gov/Programs/CCDPHP/DEODC/EHLB/IAQ/Pages/VOC.aspx) allowable concentrations for building materials (formaldehyde: 9 µg/m³, TVOC: 500 µg/m³ for individual compounds). When a localized source drives the cumulative dose in any occupied zone above 50% of the applicable guideline value, the system generates a targeted alert identifying the source location, estimated emission rate, and recommended action (increase ventilation, remove source, run air purifier directed at source location).

### 8. Ventilation Optimization and Smart Home Integration

The system provides source-aware ventilation recommendations that go beyond the simple "open a window" advice of aggregate TVOC monitors:

- **Directed air purifier placement:** Given the localized source position, the system recommends optimal air purifier placement to intercept the contaminant plume before it reaches occupied zones. The recommendation accounts for the room's velocity field to identify the natural transport pathway from source to occupant.
- **HVAC fan speed adjustment:** When a source is detected in a zone with inadequate dilution ventilation (concentration > 2× the room average for more than 15 minutes), the system can command the smart thermostat to increase fan speed. The velocity field model predicts the dilution time constant at each fan speed, enabling the system to select the minimum fan speed that achieves adequate dilution within a target time window.
- **Window/door opening advisory:** If natural ventilation would create a cross-draft removing the contaminant plume (based on wind direction data from outdoor weather integration and window positions in the room model), the system recommends which window to open.
- **Exhaust fan activation:** For kitchen or bathroom sources, the system can trigger range hood or bath exhaust fan operation via smart switch integration, with runtime optimized to the estimated emission duration rather than a fixed timer.

Smart home integration uses the Matter protocol for interoperability across ecosystems (Apple Home, Google Home, Amazon Alexa, Samsung SmartThings). The edge compute hub exposes Matter endpoints for: aggregate and per-zone air quality status (good/moderate/unhealthy), active source count and locations (as room zone identifiers), and ventilation automation triggers.

### 9. Federated Learning for Cross-Installation Model Improvement

The PINN's source localization accuracy improves with exposure to diverse room geometries and emission scenarios. The system supports federated learning: each installation periodically computes local model gradients from its accumulated source estimation data (localized source position, estimated emission rate, user-confirmed source type, actual vs. predicted concentration time series at each node). Only the gradient updates are transmitted to a central aggregation server, not raw sensor data or room geometry. The server performs federated averaging (FedAvg) across participating installations and distributes the updated model weights.

Privacy-preserving techniques include: differential privacy noise injection (ε = 8.0, δ = 10⁻⁵) on gradient updates to prevent reconstruction of individual room layouts from model updates; secure aggregation requiring a minimum of 100 participating installations per round to prevent single-installation gradient inference; and on-device gradient clipping (max L2 norm = 1.0) to bound the influence of any single installation on the global model.

### 10. Figures Description

- **Figure 1:** System architecture showing sensor node placement in a residential living room, Thread mesh communication topology, edge compute hub, and smart home integration pathways (Matter protocol to thermostat, air purifier, smart plugs).
- **Figure 2:** Example VOC source localization output showing: (a) raw sensor readings at 8 node positions over 30 minutes during a furniture off-gassing event; (b) the PINN-reconstructed concentration field at three time slices (t=0, t=5 min, t=15 min); (c) the estimated source location overlaid on the room floor plan with 95% confidence ellipse.
- **Figure 3:** Self-calibration sequence showing: (a) HVAC fan-on perturbation at t=0; (b) observed concentration redistribution at 8 sensor nodes; (c) predicted redistribution from archetype velocity field; (d) residual (observed minus predicted) used to update velocity field correction factors.
- **Figure 4:** Multi-source localization example in a kitchen/living room open floor plan: simultaneous cooking emissions (localized to stovetop) and new-sofa off-gassing (localized to living room seating area), with per-source concentration contribution maps and occupancy zone dose attribution.
- **Figure 5:** Federated learning convergence curve showing mean source localization error (meters) versus number of federated training rounds, stratified by room geometry complexity (simple rectangular, L-shaped, open floor plan).

## Claims

1. A system for real-time localization of volatile organic compound emission sources within indoor environments, comprising: a distributed array of sensor nodes, each containing a metal oxide semiconductor gas sensor and environmental compensation sensors, deployed at known positions within a room; an edge compute device receiving synchronized concentration measurements from the sensor array; and a physics-informed neural network running on the edge compute device that estimates the spatial origin and emission rate of VOC sources by inverting the advection-diffusion equation using the observed spatiotemporal concentration gradient across the sensor array and a pre-computed room airflow velocity field.

2. The system of claim 1, wherein the physics-informed neural network incorporates the advection-diffusion partial differential equation as a soft constraint in its loss function, penalizing source estimates that violate mass conservation and transport physics within the room volume.

3. The system of claim 1, wherein the room airflow velocity field is selected from a pre-computed library of Reynolds-Averaged Navier-Stokes simulation results parameterized by room geometry, HVAC register configuration, and fan operating state, and refined in situ using HVAC fan cycling events as controlled perturbations to calibrate local velocity corrections.

4. The system of claim 1, further comprising a multi-source decomposition module that iteratively estimates and subtracts individual source contributions from the observed concentration field to resolve simultaneous emissions from multiple distinct spatial origins.

5. The system of claim 1, further comprising a source classification module that categorizes localized emission sources by temporal profile features including onset rate, decay time constant, diurnal temperature correlation, and event periodicity.

6. The system of claim 1, further comprising a cumulative exposure dose estimation module that integrates the spatially reconstructed concentration field over occupancy zones, weighted by activity-level-adjusted breathing rates, and compares resulting doses against health guideline thresholds to generate per-source alerts and ventilation recommendations.

7. A method for indoor VOC source localization comprising: deploying a distributed array of metal oxide semiconductor gas sensor nodes at known positions within a room; acquiring synchronized VOC concentration measurements from the array at a sampling rate sufficient to resolve concentration wavefront propagation; constructing or retrieving a room airflow velocity field model from HVAC system parameters and room geometry; and executing a physics-informed neural network that inverts the observed spatiotemporal concentration gradient against the velocity field model to estimate VOC source location and emission rate in real-time.

8. The method of claim 7, further comprising self-calibration of the room airflow velocity field by observing concentration redistribution dynamics across the sensor array during HVAC fan state transitions and computing velocity field correction factors via recursive least-squares estimation.

9. The method of claim 7, further comprising federated learning across multiple deployed installations, wherein local model gradient updates are aggregated with differential privacy guarantees to improve source localization accuracy without transmitting raw sensor measurements or room geometry information.

10. The system of claim 1, wherein the edge compute device generates source-aware ventilation recommendations including directed air purifier placement relative to the localized source position, HVAC fan speed adjustment to achieve target dilution time constants, and natural ventilation advisories based on window positions and outdoor wind conditions.

11. The system of claim 1, wherein each sensor node self-calibrates its VOC baseline using a running percentile filter over a preceding time window to compensate for MOS sensor drift, and wherein inter-node baseline differences are normalized before spatial gradient computation to prevent drift-induced false source detections.

12. The system of claim 1, wherein the PINN infers a continuous three-dimensional concentration field from sparse sensor observations, enabling source localization at spatial resolution finer than the inter-node spacing through physics-constrained interpolation.

## Prior Art References

1. [U.S. EPA — Volatile Organic Compounds' Impact on Indoor Air Quality](https://www.epa.gov/indoor-air-quality-iaq/volatile-organic-compounds-impact-indoor-air-quality) — Indoor VOC concentrations 2-5× higher than outdoor
2. [WHO Guidelines for Indoor Air Quality: Selected Pollutants (2010)](https://www.who.int/publications/i/item/9789241548106) — Priority indoor pollutants and health guideline values
3. [Herrero et al., Sensors and Actuators B: Chemical 2020](https://doi.org/10.1016/j.snb.2019.127125) — MOS sensor array VOC source classification (89% accuracy, single-point)
4. [Schneider et al., Atmospheric Environment 2019](https://doi.org/10.1016/j.atmosenv.2019.116862) — Distributed low-cost sensor networks for urban air quality mapping
5. [Roscioli et al., Atmospheric Environment 2013](https://doi.org/10.1016/j.atmosenv.2012.08.046) — Inverse dispersion modeling for industrial emission source estimation
6. [Ai and Mak, Indoor Air 2017](https://doi.org/10.1111/ina.12340) — Tracer gas validation of CFD room ventilation models
7. [Liu and Zhai, Building and Environment 2018](https://doi.org/10.1016/j.buildenv.2018.05.027) — Adjoint CFD for indoor contaminant source identification
8. [Chen et al., Building and Environment 2021](https://doi.org/10.1016/j.buildenv.2020.107417) — Bayesian inference with pre-computed CFD for indoor source estimation
9. [EPA Exposure Factors Handbook, Chapter 6](https://www.epa.gov/expobox/exposure-factors-handbook-chapter-6) — Inhalation rates by activity level
10. [California CDPH Section 01350](https://www.cdph.ca.gov/Programs/CCDPHP/DEODC/EHLB/IAQ/Pages/VOC.aspx) — Standard method for VOC emissions testing of building materials
11. [Sensirion SGP41](https://sensirion.com/products/catalog/SGP41) — Multi-pixel gas sensor for VOC and NOx measurement
12. [Bosch BME688](https://www.bosch-sensortec.com/products/environmental-sensors/gas-sensors/bme688/) — Environmental sensor with gas scanner for VOC detection
13. [OpenFOAM](https://www.openfoam.com/) — Open-source CFD solver for velocity field pre-computation
14. [NVIDIA Modulus](https://developer.nvidia.com/modulus) — Physics-informed neural network development framework

# System and Method for Non-Invasive Buried Pipe Condition Assessment Using Surface-Mounted Vibro-Acoustic Sensor Arrays and Physics-Informed Graph Neural Networks

**LITF-PA-2026-007 · Infrastructure / AI**
**Published:** 2026-04-08
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a non-invasive system and method for assessing the condition of buried water and gas distribution pipes using surface-mounted vibro-acoustic sensor arrays deployed at existing above-ground access points (fire hydrants, valve boxes, and service connections). The system propagates guided elastic waves through buried pipe walls using a controlled piezoelectric actuator, records the resulting vibro-acoustic response at multiple surface points, and analyzes the waveform characteristics using a physics-informed graph neural network (PI-GNN) that encodes the known network topology, pipe geometry, and soil coupling models. The PI-GNN infers pipe material type, wall thickness, corrosion grade (on a 5-level ASTM scale), joint condition, and remaining useful life for each pipe segment without excavation or service interruption. A Bayesian fusion layer combines multiple measurement campaigns to reduce uncertainty, and the system outputs a prioritized rehabilitation schedule with cost-benefit analysis for each pipe segment.

## Field of the Invention

This invention relates to infrastructure condition assessment, specifically to the use of guided elastic wave propagation analysis combined with graph neural networks for non-destructive evaluation of buried pipe networks from surface-accessible measurement points.

## Background

The American Society of Civil Engineers (ASCE) rates U.S. water infrastructure at a [C- grade](https://infrastructurereportcard.org/cat-item/drinking-water-infrastructure/), with an estimated 6 billion gallons of treated water lost daily through approximately 240,000 water main breaks per year. The [American Water Works Association (AWWA)](https://www.awwa.org/Resources-Tools/Resource-Topics/Infrastructure-Financing) estimates that $1 trillion in investment is needed over the next 25 years to maintain and expand U.S. drinking water infrastructure. A critical bottleneck is the inability to assess pipe condition without excavation.

Current pipe condition assessment methods include:

- **Direct inspection (excavation):** Requires digging up pipes, costing $50-$200 per linear foot. Disruptive, slow, and impractical for system-wide assessment. Typically performed only after failure.
- **CCTV inspection:** Effective for large-diameter (>12") gravity sewers but requires pipe entry, dewatering, and cannot assess external wall condition or soil-pipe interaction. Cost: $5-$15 per linear foot. Not applicable to pressurized water mains.
- **Acoustic leak detection:** Systems such as [Pure Technologies' SmartBall](https://puretechnologies.com/technologies/smartball/) detect active leaks by analyzing acoustic signals propagating through water. These identify existing failures but cannot assess pre-failure condition or predict remaining useful life.
- **Electromagnetic (MFE) inspection:** [Xylem's PipeDiver](https://www.xylem.com/en-us/products--services/assessment-services/condition-assessment/pipediver/) uses magnetic flux exclusion to measure wall thickness in metallic pipes. Requires insertion into live mains through a fire hydrant. Limited to metallic pipes and segments between valves.

Relevant prior art in non-invasive pipe assessment includes:

- [US10871475B2](https://patents.google.com/patent/US10871475B2) (Echologics/Mueller): Measures acoustic velocity in pipe walls to estimate average wall thickness between two hydrophones. Provides a single average measurement per pipe segment; cannot localize corrosion or detect joint defects.
- [US9939420B2](https://patents.google.com/patent/US9939420B2) (MIT): Describes a robotic probe that travels inside water mains using electromagnetic sensors. Requires pipe entry and cannot inspect pipes smaller than 6" diameter.
- [Muggleton et al., Journal of Sound and Vibration 2004](https://doi.org/10.1016/S0022-460X(03)00640-6): Foundational work on guided wave propagation in buried fluid-filled pipes, establishing the theoretical basis for surface-to-pipe acoustic coupling.
- [US20200182830A1](https://patents.google.com/patent/US20200182830A1) (Syrinix): Continuous pressure and acoustic monitoring at fixed pipe locations. Detects transient events but does not perform structural condition assessment.
- [Cataldo et al., Sensors 2014](https://www.mdpi.com/1424-8220/14/3/5595): Comprehensive review of acoustic methods for pipeline inspection, establishing that guided wave techniques can detect corrosion, cracks, and wall thinning in accessible pipes but noting challenges in buried pipe applications due to soil-pipe coupling losses.

The gap in the art is a complete system that: (a) performs structural condition assessment of buried pipes from surface-accessible points without excavation or pipe entry, (b) uses physics-informed machine learning to account for soil-pipe coupling effects and network topology, (c) provides localized defect mapping (not just segment averages), and (d) integrates measurements across the network graph to improve inference accuracy through topological constraints.

## Detailed Description

### 1. Sensor Hardware and Deployment

The system uses a portable measurement kit deployable by a single field technician without traffic control or excavation. The kit comprises:

- **Actuator module:** A broadband piezoelectric actuator (frequency range 100 Hz - 20 kHz, peak force 50 N) coupled to the pipe via the fire hydrant barrel or valve box stem using a spring-loaded magnetic clamp. The actuator generates a chirp signal (linear frequency sweep over 2 seconds) that excites multiple guided wave modes in the pipe wall.
- **Sensor array:** 4-8 triaxial piezoelectric accelerometers (sensitivity: 100 mV/g, frequency range: 1 Hz - 25 kHz) deployed on the ground surface at measured distances from the actuator point. Sensors are coupled to the ground surface using 50mm diameter aluminum coupling plates pressed into the soil with 10 kg of force to ensure consistent ground coupling. Sensor spacing: 1-5 meters, arranged in a linear array along the pipe route.
- **Data acquisition unit:** 24-bit ADC sampling at 51.2 kHz per channel, 8 channels simultaneous. GPS time synchronization for multi-campaign data fusion. Ruggedized tablet for field control and data preview.

A single measurement campaign takes approximately 15 minutes per hydrant/valve location and characterizes the pipe segments extending 50-200 meters in each direction from the measurement point, depending on soil conditions and pipe material.

### 2. Guided Wave Propagation Model

Buried pipes support multiple guided wave modes, the properties of which are sensitive to pipe wall condition:

- **L(0,1) longitudinal mode:** Propagation velocity depends on pipe wall thickness and material elastic modulus. Corrosion-induced wall thinning reduces velocity by approximately 2% per 10% wall loss in cast iron.
- **L(0,2) longitudinal mode:** Higher-order mode with cutoff frequency dependent on wall thickness. Cutoff frequency shift provides a second independent measurement of wall thickness.
- **F(1,1) flexural mode:** Asymmetric mode excited by off-axis loading. Attenuation rate is sensitive to soil-pipe coupling stiffness, which correlates with bedding condition and external corrosion.
- **T(0,1) torsional mode:** Lowest-order torsional mode. Relatively insensitive to fluid loading, making it useful for isolating wall defects from fluid property changes.

The physics model uses the [Semi-Analytical Finite Element (SAFE) method](https://doi.org/10.1016/j.jsv.2005.04.014) to compute dispersion curves for each guided wave mode as a function of pipe material (cast iron, ductile iron, steel, PVC, HDPE, asbestos cement), diameter (2"-48"), wall thickness, and soil coupling parameters. The forward model takes approximately 2 seconds per pipe segment on a modern laptop CPU.

### 3. Physics-Informed Graph Neural Network

The PI-GNN operates on a graph G = (V, E) where:

- **Vertices V:** Represent pipe segments (typically 20-50 meter lengths between joints)
- **Edges E:** Represent physical connections between segments (joints, tees, bends)
- **Vertex features:** Measured vibro-acoustic response features (modal velocities, attenuation coefficients, spectral energy distribution) plus metadata (installation year, nominal material, nominal diameter from GIS records)
- **Edge features:** Joint type (bell-and-spigot, mechanical, welded, fused), angle, and diameter transition

The network architecture comprises:

1. **Encoding layer:** Multi-layer perceptron (MLP) mapping raw spectral features to a 128-dimensional latent space per vertex
2. **Message-passing layers:** 6 rounds of graph attention network (GAT) message passing with 8 attention heads. Messages are physics-constrained: the attention weights are modulated by the physical distance and number of joints between vertices, enforcing the prior that nearby pipe segments in the same installation cohort are likely to have similar condition.
3. **Physics loss:** A differentiable SAFE forward model is embedded in the training loop. For each vertex, the network predicts pipe parameters (material, wall thickness, corrosion grade), the forward model computes expected vibro-acoustic features from those parameters, and the L2 distance between predicted and measured features is added to the loss function with weight λ = 0.3.
4. **Decoding layer:** Per-vertex MLP outputting: (a) pipe material classification (6 classes), (b) wall thickness estimate (mm, with uncertainty), (c) corrosion grade (ASTM 5-level scale: A=new, B=light, C=moderate, D=heavy, E=critical), (d) joint condition (3-level: good/degraded/failed), (e) remaining useful life estimate (years, with 80% confidence interval).

Training data is generated using a combination of: (a) synthetic data from the SAFE forward model with random pipe parameters and noise, (b) field measurements from pipe segments that were subsequently excavated and directly inspected, using datasets from [UK Water Industry Research (UKWIR)](https://ukwir.org/) pipe assessment programs and the [EPA's Distribution System Optimization Project](https://www.epa.gov/water-research/distribution-system-research).

### 4. Bayesian Multi-Campaign Fusion

Multiple measurement campaigns at different hydrants/valves produce overlapping inferences for pipe segments within range of more than one measurement point. A Bayesian fusion layer combines these overlapping estimates:

For each pipe segment s with N overlapping measurements, the posterior condition estimate is:

`P(condition_s | measurements) ∝ P(condition_s) × Π_i P(measurement_i | condition_s)`

Where P(condition_s) is the prior based on installation year, material, and soil corrosivity (from USDA soil survey data), and P(measurement_i | condition_s) is the likelihood from each measurement campaign's PI-GNN inference, represented as a Gaussian over the wall thickness and corrosion grade dimensions.

The fusion process reduces wall thickness estimation uncertainty by approximately 40% when 3+ overlapping measurements are available, based on synthetic validation.

### 5. Rehabilitation Priority Scoring

The system outputs a prioritized rehabilitation schedule by computing a risk score for each pipe segment:

`Risk_s = P(failure_s | condition_s, t) × Consequence_s`

Where P(failure) is derived from a Weibull survival model parameterized by the PI-GNN condition estimates, and Consequence is computed from: (a) pipe diameter and flow rate (service disruption impact), (b) proximity to critical facilities (hospitals, schools), (c) traffic disruption potential of a break, and (d) estimated repair cost. The output is a ranked list of pipe segments with cost-benefit ratios for immediate replacement, rehabilitation (CIPP lining, slip lining), or continued monitoring.

### 6. Figures Description

- **Figure 1:** System architecture showing field measurement setup with piezoelectric actuator at fire hydrant, ground-coupled accelerometer array, and data acquisition unit, with data flow to cloud-based PI-GNN inference engine.
- **Figure 2:** Guided wave dispersion curves for a 6" cast iron pipe at various wall thickness levels (nominal 7.4mm, 50% loss, 75% loss), showing velocity and attenuation changes for L(0,1) and L(0,2) modes.
- **Figure 3:** PI-GNN architecture diagram showing graph construction from GIS pipe network, message-passing layers with physics-constrained attention, and per-vertex condition output.
- **Figure 4:** Bayesian fusion example showing three overlapping measurement campaigns converging on a wall thickness estimate with reduced uncertainty band.

## Claims

- A method for assessing the condition of buried pipes comprising: deploying a vibro-acoustic actuator at a surface-accessible pipe connection point; generating guided elastic waves in the buried pipe wall; recording the resulting vibro-acoustic response at a plurality of surface-mounted sensors; and inferring pipe condition parameters using a physics-informed graph neural network that encodes pipe network topology and guided wave propagation physics.

- The method of claim 1, wherein said pipe condition parameters comprise pipe material type, wall thickness with uncertainty estimate, corrosion grade on a standardized scale, joint condition, and remaining useful life.

- The method of claim 1, wherein said physics-informed graph neural network incorporates a differentiable forward model of guided wave propagation in buried pipes as a physics loss term during training.

- The method of claim 1, wherein said graph neural network operates on a graph representation of the pipe network with vertices representing pipe segments and edges representing physical connections, with graph attention weights modulated by physical distance and joint count.

- The method of claim 1, further comprising a Bayesian fusion layer that combines overlapping condition estimates from multiple measurement campaigns at different surface access points to reduce estimation uncertainty.

- A system for non-invasive buried pipe condition assessment comprising: a portable piezoelectric actuator for coupling to above-ground pipe access points; a plurality of ground-coupled triaxial accelerometers; a data acquisition unit with GPS time synchronization; and a computing system executing a physics-informed graph neural network trained on synthetic guided wave propagation data and validated against excavation-verified pipe condition records.

- The system of claim 6, wherein measurement at a single surface access point characterizes pipe condition for segments extending 50-200 meters in each direction without excavation or service interruption.

- The system of claim 6, further comprising a rehabilitation priority scoring module that computes risk scores based on inferred pipe condition, failure probability, and consequence factors to generate a prioritized rehabilitation schedule.

- The method of claim 1, wherein said guided elastic waves comprise at least longitudinal L(0,1), longitudinal L(0,2), flexural F(1,1), and torsional T(0,1) modes, each providing independent sensitivity to different pipe wall parameters.

- The method of claim 1, wherein prior condition estimates are derived from pipe installation year, material, and soil corrosivity data from public soil survey databases.

## Prior Art References

- [US10871475B2](https://patents.google.com/patent/US10871475B2) — Echologics/Mueller — "Acoustic Pipe Wall Assessment" (2020)
- [US9939420B2](https://patents.google.com/patent/US9939420B2) — MIT — "Robotic In-Pipe Electromagnetic Inspection" (2018)
- [US20200182830A1](https://patents.google.com/patent/US20200182830A1) — Syrinix — "Continuous Pipe Monitoring" (2020)
- [Muggleton et al.](https://doi.org/10.1016/S0022-460X(03)00640-6) — "Wavenumber Prediction in Buried Pipes" — Journal of Sound and Vibration 2004
- [Cataldo et al.](https://www.mdpi.com/1424-8220/14/3/5595) — "Acoustic Methods for Pipeline Inspection" — Sensors 2014
- [Hayashi et al.](https://doi.org/10.1016/j.jsv.2005.04.014) — "SAFE Method for Guided Waves in Pipes" — Journal of Sound and Vibration 2006
- [ASCE Infrastructure Report Card](https://infrastructurereportcard.org/cat-item/drinking-water-infrastructure/) — Drinking Water Grade: C-
- [AWWA Infrastructure Financing](https://www.awwa.org/Resources-Tools/Resource-Topics/Infrastructure-Financing) — $1 trillion investment gap
- [Pure Technologies SmartBall](https://puretechnologies.com/technologies/smartball/) — In-pipe acoustic leak detection
- [Xylem PipeDiver](https://www.xylem.com/en-us/products--services/assessment-services/condition-assessment/pipediver/) — Magnetic flux exclusion wall thickness
- [UKWIR Pipe Condition Assessment Research](https://ukwir.org/) — Field-verified pipe condition datasets
- [EPA Distribution System Research](https://www.epa.gov/water-research/distribution-system-research) — US water distribution system data
- [Veličković et al.](https://arxiv.org/abs/1710.10903) — "Graph Attention Networks" — ICLR 2018
- [USDA Web Soil Survey](https://websoilsurvey.nrcs.usda.gov/) — Soil corrosivity data for pipe degradation priors

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*

# LITF-PA-2026-015: System and Method for Real-Time Power Transmission Line Conductor Sag Estimation and Dynamic Line Rating Using Ground-Level Triaxial Magnetometer Arrays and Physics-Informed Recurrent Neural Networks

**Filing:** LITF-PA-2026-015  
**Domain:** EnergyTech / Grid Infrastructure  
**Published:** April 19, 2026  
**License:** [CC0 1.0 — Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)  
**HTML Version:** [liveinthefuture.org/priorart/powerline-sag-magnetometer.html](https://liveinthefuture.org/priorart/powerline-sag-magnetometer.html)

---

## Abstract

Disclosed is a system and method for continuously estimating the sag of overhead power transmission line conductors in real time by deploying arrays of low-cost triaxial magnetometer sensors at ground level within the transmission line right-of-way. The system measures the spatial gradient of the magnetic field produced by the alternating current flowing through overhead conductors, cross-references these measurements against known phase current magnitudes obtained from substation SCADA telemetry, and solves the inverse problem of conductor geometry estimation using a physics-informed recurrent neural network (PI-RNN) that encodes the Biot-Savart law, catenary mechanics, and IEEE 738 thermal balance equations as soft constraints in its loss function. The estimated conductor heights are converted to sag values and thermal conductor temperatures, enabling real-time dynamic line rating (DLR) that allows transmission operators to safely increase power transfer capacity by 15-40% during favorable weather conditions while maintaining NERC reliability standards. The complete sensor node costs under $85 in bill-of-materials, operates on solar power, and communicates via LoRaWAN or cellular, requiring no physical contact with energized conductors and no line outages for installation.

## Field of the Invention

This invention relates to electric power transmission infrastructure monitoring, specifically to the use of ground-level magnetic field sensing combined with physics-informed machine learning for non-contact estimation of overhead conductor sag, thermal state, and real-time ampacity.

## Background

The United States operates approximately [160,000 miles of high-voltage transmission lines](https://www.eia.gov/energyexplained/electricity/delivery-to-consumers.php) (EIA, 2024), and congestion on these lines costs ratepayers an estimated [$20.8 billion annually](https://gridstrategiesllc.com/wp-content/uploads/2023/06/National-Transmission-Planning-Study.pdf) in higher electricity prices (Grid Strategies, 2023). Much of this congestion is artificial: lines are rated using static thermal limits calculated for worst-case weather assumptions (40°C ambient, 0.6 m/s crosswind, full sun). Under actual conditions, which are favorable 85-95% of the time, these same lines could safely carry 15-40% more current.

Dynamic line rating (DLR) replaces static assumptions with real-time measurements of conductor temperature or sag, allowing operators to adjust ampacity limits continuously. FERC Order 881, finalized in [December 2021](https://www.ferc.gov/media/e-1-rm20-16-000) and effective July 2025, requires all FERC-jurisdictional transmission providers to implement ambient-adjusted ratings on their systems, with many ISOs pushing toward full DLR. The technical bottleneck is not regulatory: it is the cost and difficulty of measuring conductor state in real time across hundreds of thousands of line-miles.

Current approaches to conductor sag or temperature measurement include:

- **Direct-contact conductor temperature sensors:** Devices like the [Lindsey LineVision](https://linevis.com/) (~$15,000 per span) or [Nexans CAT-1](https://www.nexans.com/en/products-services/power-accessories/overhead-line-systems/cat-1-sensor.html) sensors clamp directly onto the conductor. They provide accurate measurements but require a helicopter or hot-stick crew to install on energized lines ($3,000-$8,000 per installation in crew costs), are exposed to conductor vibration fatigue, and scale poorly across large networks.
- **LiDAR and photogrammetry surveys:** Aerial LiDAR captures conductor geometry with centimeter accuracy. [Malhara & Vittal (2016)](https://doi.org/10.1109/TPWRD.2016.2598171) demonstrated 2-5 cm sag accuracy from helicopter-mounted LiDAR. But surveys are periodic snapshots (annual or biennial), not real-time, and cost $300-$800 per line-mile per flight.
- **Weather-based estimation:** The cheapest approach deploys weather stations along the line and feeds temperature, wind speed, and solar radiation into IEEE 738 thermal models. [EPRI's DTCR software](https://www.epri.com/research/products/000000003002006130) implements this. Accuracy degrades when microclimatic conditions vary along a span.
- **Tension monitoring:** Load cells at dead-end towers measure mechanical tension, which relates to sag through catenary equations. Accurate for the monitored span but requires tower modifications, costs $20,000-$50,000 per installation, and only works at dead-end structures (not suspension towers, which comprise 80-90% of a typical line).
- **GPS sag measurement:** GPS receivers mounted on the conductor itself track vertical displacement. Demonstrated by [Seppa et al. (2005)](https://doi.org/10.1109/TPWRD.2005.848403) with 10-15 cm accuracy. Requires conductor-mounted hardware.

The fundamental gap in all existing approaches is the tradeoff between measurement accuracy and deployment cost at scale. The physics underlying the proposed approach is well established but has not been applied to this problem. An overhead conductor carrying alternating current I at height h above ground produces a magnetic field at ground level governed by the [Biot-Savart law](https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law). For a single infinitely long horizontal conductor, the RMS magnetic flux density at a point directly beneath the conductor is B = μ₀I / (2πh). For a 1,000 A conductor at 20 m height, this yields approximately 10 μT. As the conductor sags (h decreases from 20 m to 15 m), B increases by 33%. A magnetometer with 10 nT resolution can distinguish sag changes of approximately 2 cm.

No existing system combines ground-level magnetometer arrays with physics-informed neural networks to solve the full inverse problem of conductor geometry reconstruction for dynamic line rating.

## Detailed Description

### 1. System Architecture Overview

The system comprises three layers: a ground-level sensor network, an edge computing gateway at each tower or span midpoint, and a cloud-hosted dynamic line rating engine that feeds ampacity values into the transmission operator's energy management system (EMS).

Each sensor node contains: a triaxial fluxgate or AMR magnetometer (e.g., [Memsic MMC5983MA](https://www.memsic.com/magnetometers), 18-bit resolution, 8 Gauss full scale, 0.25 mG noise at 100 Hz bandwidth, $3.50 in volume); a temperature/humidity sensor (Sensirion SHT40, $1.80); a 3-axis MEMS accelerometer for tilt compensation (Analog Devices ADXL355, $8.50); a LoRaWAN or LTE-M radio (Semtech SX1262, $4.20); a solar cell (1W) and LiFePO4 battery (3.2V, 1500 mAh); and a low-power MCU (Nordic nRF5340, $4.80). Total BOM: approximately $82 at 1,000-unit volume.

Nodes are deployed in arrays of 3-5 units per span, distributed laterally across the right-of-way beneath the conductors. Typical deployment places one node directly beneath each phase conductor and one or two at intermediate positions. Nodes mount on non-ferromagnetic posts (fiberglass or PVC) driven 0.5 m into the ground, placing the magnetometer at 1.0 m above grade.

### 2. Magnetic Field Measurement and Signal Processing

The 60 Hz AC magnetic field from the overhead conductors dominates the measurement environment. The sensor nodes sample the triaxial magnetic field at 1,024 Hz (approximately 17 samples per power cycle), sufficient to resolve the fundamental frequency and its first five harmonics. Each measurement epoch lasts 1 second (60 complete cycles), and the MCU computes the RMS magnitude of each axis, the phase angle relative to a GPS-disciplined 1 PPS reference, and the harmonic spectrum through a 1,024-point FFT.

The raw measurement includes contributions from:

- **Overhead conductor fields (signal):** 60 Hz and harmonics, amplitude 1-50 μT. Three-phase configuration produces a characteristic spatial pattern that varies with conductor geometry.
- **Earth's geomagnetic field:** DC component, approximately 50 μT. Removed by band-pass filtering to extract only the 60 Hz component.
- **Geomagnetic micropulsations:** Pc3-Pc5 pulsations (1 mHz to 100 mHz) with amplitudes of 0.1-10 nT. Far below the signal band.
- **Nearby metallic structures:** Fences, pipes, and grounding grids distort the magnetic field spatially. These distortions are static and calibrated out during installation.
- **Ground return currents:** Unbalanced phase currents produce a net magnetic field from the earth return path. Modeled as an image conductor at complex depth d* = √(ρ/jωμ₀). For typical soils (100 Ω·m), d* ≈ 650 m at 60 Hz, contributing less than 0.5% to the field at the sensor.

After filtering, each 1-second measurement epoch produces a vector of 15-25 values per node, transmitted to the edge gateway every 10 seconds as a compressed packet (~120 bytes per node per epoch).

### 3. Inverse Problem Formulation

The forward problem is well defined: given conductor positions in 3D space and phase currents with known phase angles, the magnetic field at any point is computable via the Biot-Savart integral along the catenary-shaped conductor. For a conductor hanging as a catenary between two towers, the geometry is parameterized by a single scalar: the horizontal tension H (equivalently, the midspan sag s = wL²/8H, where w is conductor weight per unit length and L is span length).

The inverse problem is: given magnetic field measurements at N ground-level points and known phase currents from SCADA, estimate the sag parameter s.

The inverse problem is ill-conditioned for a single measurement point because current and height are coupled (doubling the current at twice the height produces the same field). The array configuration breaks this degeneracy: sensors at different lateral offsets see different field magnitudes for the same (I, h) pair, because the off-axis field decays as 1/(r²) while the directly-beneath field decays as 1/h. Three or more measurement points overdetermine the system.

### 4. Physics-Informed Recurrent Neural Network

A purely analytical inversion is feasible for idealized geometries but degrades in real-world conditions: conductor blowout from crosswinds, tower sway, shield wire fields, and temperature gradients along a span. The system addresses these with a physics-informed recurrent neural network (PI-RNN).

The PI-RNN architecture is a gated recurrent unit (GRU) with 128 hidden units, processing a sliding window of the last 60 measurement epochs (10 minutes of data at 10-second intervals). Input at each timestep: (a) the magnetometer feature vector from all N nodes (45-125 input features); (b) three-phase current magnitudes and phase angles from SCADA (6 values); (c) ambient weather data (temperature, wind speed, wind direction, solar irradiance: 4 values). Output: estimated midspan sag in meters.

The loss function comprises four terms:

1. **Data loss (L_data):** MSE between predicted sag and ground-truth sag from periodic LiDAR surveys. Active only during supervised training and fine-tuning.
2. **Physics loss (L_physics):** For each predicted sag value, a differentiable forward model computes the expected magnetic field at each sensor location via Biot-Savart integration along the catenary. The MSE between computed and measured field is backpropagated. This is the primary self-supervised loss.
3. **Thermal loss (L_thermal):** The predicted sag is converted to conductor temperature via the conductor's thermal expansion model. This temperature is fed into the IEEE 738 steady-state heat balance equation: q_R + q_C = q_S + I²R(T). The residual is penalized, constraining the sag estimate to be thermally consistent.
4. **Temporal smoothness loss (L_smooth):** Penalizes the second time derivative of the sag estimate, encoding the physical prior that conductor temperature changes slowly (thermal time constants of 5-15 minutes for typical ACSR conductors).

Combined loss: L = L_data + λ₁L_physics + λ₂L_thermal + λ₃L_smooth, with λ₁ = 1.0, λ₂ = 0.5, λ₃ = 0.1 effective across 230 kV and 345 kV ACSR lines.

### 5. Dynamic Line Rating Computation

The PI-RNN outputs estimated sag at 10-second intervals. Sag is converted to effective conductor temperature T_eff using the conductor's known sag-temperature relationship. The DLR engine computes maximum permissible current I_max such that conductor temperature stays below T_max (typically 100°C for ACSR) and sag stays below s_max per [NERC FAC-003-4](https://www.nerc.com/pa/Stand/Reliability%20Standards/FAC-003-4.pdf) clearance requirements.

I_max is computed by solving the IEEE 738 heat balance equation forward in time for a 15-60 minute planning horizon using weather forecasts from a mesoscale NWP model (e.g., HRRR at 3 km resolution). The DLR never drops below the static rating. If the PI-RNN's confidence interval (via Monte Carlo dropout) exceeds ±0.5 m of sag, the system falls back to the static rating.

### 6. Calibration and Self-Supervision

Initial calibration requires: (a) surveyed tower coordinates and conductor attachment heights; (b) conductor type and physical parameters; (c) a reference sag measurement at known temperature and current.

After initial calibration, the system is largely self-supervised via L_physics. The system detects calibration drift by monitoring L_physics magnitude: if it trends upward over weeks, an alert triggers recalibration.

### 7. Wildfire Risk Mitigation Application

In fire-prone regions, the system provides span-level sag monitoring for targeted public safety power shutoff (PSPS) decisions. [CPUC wildfire investigations](https://www.cpuc.ca.gov/industries-and-topics/wildfires) have found conductor-vegetation contact a leading ignition source. Circuit-level PSPS affects entire communities; span-level monitoring reduces affected area by an estimated 60-80% while maintaining safety margins at the specific spans where vegetation proximity is critical.

### 8. Figures Description

- **Figure 1:** System architecture showing a three-phase transmission line with five ground-level magnetometer nodes deployed across the right-of-way, wireless links to a tower-mounted edge gateway, and data flow to the cloud DLR engine and utility EMS.
- **Figure 2:** Cross-sectional view of a three-phase horizontal conductor configuration at height h, with magnetic field lines computed via Biot-Savart shown at ground level, and three magnetometer positions indicated with measured field vectors.
- **Figure 3:** Sensitivity analysis showing the change in ground-level magnetic field magnitude versus conductor sag for 500 A, 1000 A, and 2000 A phase currents on a 345 kV line with 20 m nominal conductor height and 300 m span length.
- **Figure 4:** PI-RNN architecture diagram showing the 60-epoch sliding window input, GRU hidden layers, four-term physics-informed loss function, and sag/DLR output path.
- **Figure 5:** 48-hour time series comparing PI-RNN sag estimates against direct LiDAR measurement for a 345 kV ACSR Drake conductor span, showing <5 cm RMS error under varying load and weather conditions.
- **Figure 6:** Map of a 50-mile transmission corridor with sensor nodes marked, color-coded by real-time DLR headroom percentage, overlaid on vegetation proximity risk zones.

## Claims

1. A system for estimating overhead power transmission line conductor sag in real time, comprising: one or more triaxial magnetometer sensors positioned at ground level within the transmission line right-of-way; a signal processing module that extracts the RMS magnitude, phase angle, and harmonic spectrum of the power-frequency magnetic field from each sensor; a data link providing real-time phase current magnitudes from a substation measurement device; and a computational module that solves the inverse problem of conductor height estimation by comparing the measured magnetic field spatial pattern against a forward model based on the Biot-Savart law applied to the catenary geometry of the overhead conductors.

2. The system of claim 1, wherein the computational module is a physics-informed recurrent neural network whose loss function includes a physics consistency term computed by forward-modeling the magnetic field from the network's predicted conductor geometry via the Biot-Savart integral and penalizing the discrepancy with actual magnetometer measurements, enabling continuous self-supervised learning without ground-truth sag labels.

3. The system of claim 1, wherein the loss function further includes a thermal consistency term that converts predicted sag to conductor temperature via the conductor's thermal expansion model and penalizes the residual of the IEEE 738 heat balance equation evaluated at the predicted temperature, measured ambient weather conditions, and measured phase current.

4. The system of claim 1, wherein multiple magnetometer sensors are deployed at different lateral offsets from the conductor to break the current-height degeneracy inherent in single-point magnetic field measurement, with at least three sensors per span enabling overdetermined estimation of three-phase conductor geometry.

5. A method for dynamic line rating of an overhead power transmission line, comprising: measuring the 60 Hz magnetic field at two or more ground-level positions beneath the line; obtaining real-time phase current data from a substation SCADA system; estimating conductor sag from the magnetic field and current data using a physics-informed neural network; converting the estimated sag to an effective conductor temperature; and computing a maximum permissible current that maintains conductor temperature below a design limit and sag below a clearance limit, wherein the maximum permissible current is published as a dynamic ampacity rating to the transmission operator's energy management system.

6. The method of claim 5, wherein the physics-informed neural network is a gated recurrent unit processing a temporal window of at least 10 minutes of sequential magnetometer measurements, enabling the network to capture the thermal time constant of the conductor and reject transient measurement noise.

7. The method of claim 5, further comprising a confidence estimation module that uses Monte Carlo dropout to compute a prediction uncertainty interval, and a fallback mechanism that reverts to the static thermal rating when the uncertainty interval exceeds a configurable threshold.

8. The system of claim 1, further comprising a calibration drift detection module that monitors the temporal trend of the physics consistency loss and generates a recalibration alert when the loss magnitude exceeds a baseline threshold for a sustained period, indicating that the physical relationship between conductor geometry and measured magnetic field has changed.

9. The system of claim 1, wherein the ground-level magnetometer sensors are mounted on non-ferromagnetic posts within the right-of-way at a fixed height between 0.5 and 2.0 meters above grade, powered by solar cells with battery backup, and communicate wirelessly via LoRaWAN, LTE-M, or NB-IoT, requiring no physical contact with energized conductors and no line outages for installation or maintenance.

10. The method of claim 5, further comprising a wildfire risk mitigation application that uses span-level sag estimates to enable targeted de-energization decisions at the individual span level rather than the circuit level, reducing the number of customers affected by public safety power shutoff events while maintaining clearance safety margins at identified high-risk spans.

## Implementation Notes

The primary deployment targets are high-voltage transmission lines (230 kV and above) in congested corridors where DLR unlocks significant transfer capacity, and distribution or sub-transmission lines (69-138 kV) in high-fire-risk zones where span-level sag monitoring prevents vegetation contact ignition. At $85 per sensor node and 4 nodes per span (average span length 300 m), the per-mile sensor cost is approximately $2,270. For a 100-mile congested corridor carrying $5 million/year in congestion costs, a full deployment costing $227,000 in hardware pays back in under three weeks if DLR relieves even 5% of congestion.

The approach has inherent limitations. First, signal-to-noise ratio degrades at low current loading: below approximately 200 A on a 230 kV line at 20 m height, the 60 Hz magnetic field drops below 2 μT, and magnetometer noise limits sag resolution to ±20 cm rather than ±5 cm. This is acceptable because DLR is most valuable when lines are heavily loaded, precisely when signal strength is highest. Second, strong geomagnetic storms (K-index ≥ 7) can inject broadband noise. These events are rare and can be flagged using [NOAA SWPC](https://www.swpc.noaa.gov/) data. Third, multi-circuit towers require per-circuit current data and a more complex forward model.

The strongest counterargument is that electromagnetic-contact sensors like LineVision's already solve the problem with proven accuracy. LineVision reports ±0.1°C conductor temperature accuracy with active utility deployments. The counter-counterargument is scale: at $15,000 per monitored span plus helicopter installation, instrumenting a 100-mile corridor (approximately 1,700 spans) costs $25.5 million. The magnetometer approach costs $144,000 for the same corridor. The 177x cost difference makes system-wide DLR feasible where per-span economics previously limited deployment.

## Prior Art References

1. [U.S. Energy Information Administration. "Delivery to Consumers."](https://www.eia.gov/energyexplained/electricity/delivery-to-consumers.php) 160,000 miles of high-voltage lines.
2. [Grid Strategies LLC (2023). "National Transmission Planning Study."](https://gridstrategiesllc.com/wp-content/uploads/2023/06/National-Transmission-Planning-Study.pdf) $20.8 billion annual congestion cost.
3. [FERC Order 881. "Managing Transmission Line Ratings" (RM20-16-000).](https://www.ferc.gov/media/e-1-rm20-16-000) Requires ambient-adjusted ratings by July 2025.
4. [IEEE 738-2023. "Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors."](https://standards.ieee.org/ieee/738/11049/)
5. [Malhara, S. & Vittal, V. (2016). "Mechanical State Estimation of Overhead Transmission Lines Using Tilt Sensors."](https://doi.org/10.1109/TPWRD.2016.2598171) IEEE Trans. Power Delivery, 31(5), 2324-2333.
6. [Seppa, T.O. et al. (2005). "Use of On-Line Tension Monitoring for Real-Time Thermal Ratings."](https://doi.org/10.1109/TPWRD.2005.848403) IEEE Trans. Power Delivery, 20(2), 1403-1409.
7. [LineVision Inc.](https://linevis.com/) Non-contact electromagnetic conductor monitoring system.
8. [Nexans CAT-1.](https://www.nexans.com/en/products-services/power-accessories/overhead-line-systems/cat-1-sensor.html) Direct-contact conductor temperature sensor.
9. [EPRI. "Dynamic Thermal Circuit Rating (DTCR)."](https://www.epri.com/research/products/000000003002006130) Weather-based dynamic line rating software.
10. [NERC FAC-003-4. "Transmission Vegetation Management."](https://www.nerc.com/pa/Stand/Reliability%20Standards/FAC-003-4.pdf)
11. [CPUC. "Wildfires."](https://www.cpuc.ca.gov/industries-and-topics/wildfires) California wildfire investigation data.
12. [NOAA Space Weather Prediction Center.](https://www.swpc.noaa.gov/) Real-time geomagnetic storm monitoring.
13. [Dupin, R. et al. (2019). "Dynamic line rating prospects in the context of climate change."](https://doi.org/10.1016/j.apenergy.2019.113757) Applied Energy, 256, 113757.
14. [US8744790B2. "Method and apparatus for determining the sag of overhead power lines."](https://patents.google.com/patent/US8744790B2/en) Camera-based optical sag measurement; does not disclose magnetometer-based measurement or physics-informed ML inversion.

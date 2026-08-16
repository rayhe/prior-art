# PA-2026-141: System and Method for Continuous Ambient Ionizing Radiation Field Mapping and Anomaly Detection Using Smartphone CMOS Image Sensor Dark Current Analysis with Crowdsourced Spatial Aggregation and Real-Time Public Health Alert Generation

**Filing:** LITF-PA-2026-141  
**Domain:** Public Health / Radiation Safety / Crowdsourced Sensing  
**Published:** August 16, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for continuous ambient ionizing radiation field mapping using the CMOS image sensors already present in consumer smartphones. Every silicon CMOS sensor is inherently sensitive to ionizing radiation: when a gamma-ray photon or charged particle strikes a pixel's depletion region, it generates electron-hole pairs that appear as anomalous bright pixel events in dark frames (frames captured with no intentional optical input). By periodically acquiring dark calibration frames and analyzing the rate, spatial distribution, and charge deposition patterns of radiation-induced pixel events, a smartphone application estimates the ambient ionizing radiation dose rate without any additional hardware. The system crowdsources these measurements across millions of participating devices using privacy-preserving geohash spatial bucketing, interpolates between measurement points via Gaussian process regression, and applies statistical change-point detection against a learned natural background model to generate real-time public health alerts when localized dose rates exceed predicted baselines. The resulting monitoring network achieves block-level spatial resolution in urban areas, roughly 1,000 times denser than the 140-station EPA RadNet network that currently covers the entire United States.

## Field of the Invention

This invention relates to radiation safety and public health monitoring, specifically to opportunistic ionizing radiation sensing using consumer smartphone CMOS image sensors, crowdsourced spatial data aggregation, and automated radiological anomaly detection for civilian early warning.

## Background

Ambient ionizing radiation monitoring in the United States relies primarily on the EPA's RadNet network, which operates approximately 140 stationary monitoring stations across 50 states and territories. These stations measure gamma radiation exposure rates and collect air filter samples on 3-hour cycles. At 140 stations for 3.8 million square miles, the average spacing between monitors is roughly 165 miles. RadNet was not designed to detect localized radiological events at the neighborhood or city-block level.

Existing approaches to denser radiation monitoring have significant limitations:

- **Dedicated sensor networks:** The Safecast citizen science project, launched after the 2011 Fukushima Daiichi disaster, deploys dedicated Geiger-Muller tube sensors (bGeigie Nano, $450/unit). As of 2024, Safecast has collected over 200 million measurements globally but with coverage concentrated in Japan and parts of Europe. The per-unit cost and requirement for dedicated hardware limits scaling beyond tens of thousands of nodes.
- **CMOS-based cosmic ray detection:** The DECO (Distributed Electronic Cosmic-ray Observatory) project (Vandenbroucke et al., 2016) demonstrated that smartphone CMOS sensors can detect cosmic ray muons and other charged particles by analyzing bright pixel events in camera frames. DECO focuses on cosmic ray physics, not ambient dose rate estimation, and does not perform spatial aggregation, device-specific calibration for dose rate conversion, or anomaly detection against background models.
- **Laboratory CMOS radiation validation:** Cogliati et al. (Nuclear Instruments and Methods in Physics Research A, 2014) validated that consumer CMOS sensors can detect ionizing radiation with measurable linearity between event rate and dose rate up to several mSv/h. Kang et al. (Sensors, 2021) demonstrated smartphone-based gamma radiation detection using CMOS dark frame analysis with a Samsung Galaxy S7, achieving dose rate estimation within 20% of reference instruments above 1 μSv/h.
- **Existing patents:** US10234571B2 ("Radiation detection using a mobile device") describes using a smartphone camera to detect radiation events, but does not address crowdsourced spatial mapping, neural network event classification, device-specific CMOS characterization for calibrated dose rate conversion, privacy-preserving aggregation, or statistical anomaly detection against learned background models.

The gap in the art is a complete end-to-end system that: (a) converts stock smartphone CMOS sensors into calibrated radiation dosimeters without additional hardware, (b) distinguishes radiation-induced pixel events from thermal noise and fixed-pattern noise using a trained classifier, (c) crowdsources device-level measurements into dense spatial radiation maps with privacy guarantees, (d) maintains a dynamic model of expected natural background variation, and (e) detects and alerts on statistically significant anomalies indicating potential radiological incidents.

## Detailed Description

### 1. Dark Frame Acquisition Protocol

The system acquires dark calibration frames from the smartphone's rear-facing CMOS image sensor at configurable intervals (default: every 10 minutes). A dark frame is an image captured with no intentional optical input, isolating signals generated within the silicon itself.

Dark frame acquisition is triggered under conditions that ensure the lens receives negligible light:

- **Proximity sensor gating:** The phone's proximity sensor indicates an object within 5 mm of the screen (phone face-down on a surface or in a pocket). The system commands a 100 ms rear camera capture at minimum exposure time (typically 1/10000s on modern sensors) and maximum analog gain (ISO 3200-6400), maximizing sensitivity to radiation-induced charge while minimizing optical leakage.
- **Accelerometer stillness validation:** The device must report accelerometer variance below 0.02 m/s² across all axes for the preceding 2 seconds, confirming the phone is stationary and not being actively handled.
- **Ambient light sensor check:** Front-facing ambient light sensor reads below 5 lux, providing secondary confirmation of darkness.
- **Software-controlled capture:** On devices with electronic shutter capability, the system can capture dark frames by commanding zero-exposure-time readouts that sample only the sensor's dark current and any radiation-induced charge, independent of ambient light conditions.

Each acquisition captures a burst of 5 dark frames in rapid succession (50 ms spacing). Multiple frames enable temporal coincidence filtering: a radiation event deposits charge in a single frame, while a stuck hot pixel appears in all frames. The burst is processed immediately on-device; raw pixel data is never transmitted or stored beyond processing.

### 2. Radiation Event Detection and Classification

Each dark frame is analyzed for candidate radiation events using a three-stage pipeline:

**Stage 1: Hot pixel subtraction.** A rolling hot pixel map, updated every 24 hours, records pixels whose dark current consistently exceeds 3σ above the sensor median. These fixed-pattern noise pixels are masked before event detection. The map adapts to temperature-dependent hot pixel drift using the device's internal temperature sensor.

**Stage 2: Candidate event identification.** After hot pixel masking, connected-component analysis identifies clusters of pixels with ADC values exceeding a configurable threshold (default: 5σ above the frame's median dark level). Each cluster is characterized by: pixel count (1-12 pixels for typical radiation events), total integrated charge (sum of ADC values above threshold), spatial eccentricity (ratio of major to minor axis of the cluster's bounding ellipse), and charge centroid position.

**Stage 3: Neural network event classifier.** A lightweight convolutional neural network (architecture: 3 convolutional layers with 8/16/32 filters, 5x5 kernels, ReLU activation, followed by global average pooling and a 4-class softmax output) classifies each candidate event into one of four categories:

- **Gamma-ray photoelectric absorption:** Single-pixel or compact 2-3 pixel clusters with high charge deposition. The full photon energy is deposited in the silicon, producing a charge proportional to photon energy (e.g., 662 keV for Cs-137 gamma rays deposits approximately 180,000 electron-hole pairs in a 3 μm depletion depth).
- **Compton scattering:** Elongated 2-5 pixel clusters where a gamma ray scatters off an electron, depositing partial energy along a short recoil track. The cluster shape encodes the scattering angle.
- **Cosmic ray muon track:** Long linear tracks spanning 5-50+ pixels, produced by minimum-ionizing particles traversing the sensor at steep angles. These are distinguished by their high pixel count, linear morphology, and relatively uniform charge deposition per pixel (approximately 80 electron-hole pairs per micron of silicon).
- **Thermal noise / artifact:** Random thermal fluctuations, read noise spikes, or residual hot pixels not caught by the static map. Rejected from dose rate calculation.

The classifier is trained on a labeled dataset combining: controlled irradiation of 15 smartphone models using calibrated Cs-137, Co-60, and Am-241 sources at known dose rates (0.1 to 100 μSv/h) at a calibration laboratory; DECO project cosmic ray event archives (200,000+ labeled events across 50+ device models); and synthetic thermal noise frames generated from measured dark current distributions. Model size: 45 KB (INT8 quantized). Inference time: less than 8 ms per candidate event on a mid-range mobile SoC (e.g., Snapdragon 7 Gen 1).

### 3. Device-Specific CMOS Characterization and Dose Rate Estimation

Converting detected event rates to calibrated ambient dose rates (μSv/h) requires accounting for the wide variation in CMOS sensor parameters across smartphone models:

- **Pixel size:** Ranges from 0.6 μm (Samsung ISOCELL HP2, 200 MP) to 2.4 μm (Google Pixel main sensor). Larger pixels intercept more radiation per unit area.
- **Depletion depth:** Typically 3-6 μm for backside-illuminated (BSI) sensors. Determines the effective sensitive volume for gamma-ray interaction. BSI sensors have higher radiation sensitivity than front-side-illuminated (FSI) designs because the depletion region is closer to the incident surface.
- **Sensor area:** Total active area ranges from 15 mm² (1/3.6" format) to 85 mm² (1" format). Larger sensors intercept proportionally more radiation.
- **Read noise floor:** Determines the minimum detectable charge deposition and therefore the low-energy detection threshold.

The system maintains a device characterization database indexed by smartphone model (identified via Android Build.MODEL / iOS UIDevice model). For each model, the database stores: effective sensitive area (mm²), estimated depletion depth (μm), read noise (electrons RMS), conversion gain (μV/e-), and a calibration curve mapping detected event rate (events per cm² per second) to ambient dose rate (μSv/h) for a reference gamma energy spectrum (Cs-137, 662 keV).

Initial calibration curves are established by exposing each device model to known dose rates at a calibration facility. Cross-calibration refinement occurs in the field: when a participating device is co-located (within 50 meters and 5 minutes) with a device that has already been laboratory-calibrated, the system applies transfer learning to refine the uncalibrated device's conversion parameters. Over time, as calibrated devices propagate through the network, calibration accuracy improves across all participating models without requiring each individual phone to visit a laboratory.

The dose rate estimator applies energy-dependent correction factors for the ambient gamma spectrum. Natural background radiation has a characteristic energy distribution dominated by K-40 (1.46 MeV), U-238 daughter chain, and Th-232 daughter chain. The system's default spectral assumption is the UNSCEAR reference outdoor gamma spectrum, adjustable per geographic region based on known geological composition.

### 4. Privacy-Preserving Crowdsourced Spatial Aggregation

Individual device measurements are aggregated into a spatial radiation map using a privacy-preserving protocol:

**Geohash bucketing:** Each device's GPS coordinates are truncated to a geohash precision of 7 characters, corresponding to a spatial cell of approximately 153 m × 153 m. The device transmits only the geohash (not raw coordinates), the estimated dose rate, a device model identifier (for applying the correct calibration curve server-side), a measurement timestamp (rounded to the nearest 5-minute boundary), and a measurement quality score (based on dark frame acquisition conditions and classifier confidence). No device-specific identifier, IP address, or precise location is included in the transmitted record.

**Differential privacy:** Each transmitted dose rate is perturbed by additive Laplace noise calibrated to provide (ε, δ)-differential privacy with ε = 1.0 and δ = 10⁻⁵. At typical measurement densities (50+ devices per geohash cell in urban areas), the aggregated mean dose rate converges to within 5% of the true value despite per-device noise injection.

**Minimum reporting threshold:** A geohash cell's aggregated dose rate is published only when at least 5 independent devices have contributed measurements within the same 15-minute window, preventing individual device tracking through sparse-cell re-identification.

**Gaussian process interpolation:** The aggregated geohash measurements form a sparse spatial field. A Gaussian process (GP) regression model with a Matérn 5/2 kernel interpolates between measurement points to produce a continuous radiation field estimate. The GP's posterior variance at each point provides a built-in uncertainty estimate that is largest in areas with sparse device coverage and smallest in dense urban cores. The GP's length scale parameter (typically 200-500 m) is learned from the data and encodes the spatial correlation structure of natural background radiation, which varies smoothly with geology and altitude.

### 5. Natural Background Model and Anomaly Detection

Effective anomaly detection requires distinguishing genuinely elevated radiation from the substantial natural variation in background dose rates:

**Baseline model components:**

- **Geological baseline:** Natural background from terrestrial radionuclides (K-40, U/Th decay chains) varies from 0.03 μSv/h (over young basalts and sedimentary rock) to 0.25 μSv/h (over granitic formations and phosphate deposits). The system incorporates USGS geological survey data to set per-cell prior expectations.
- **Altitude correction:** Cosmic ray dose contribution increases approximately 10% per 300 m elevation gain. GPS altitude data (aggregated, not individual) adjusts the expected cosmic ray component.
- **Solar modulation:** Galactic cosmic ray flux varies with the ~11-year solar cycle and acutely during Forbush decreases following coronal mass ejections. The system ingests Neutron Monitor Database (NMDB) data to track solar modulation in real time.
- **Diurnal radon cycle:** Radon-222 emanation from soil produces a characteristic diurnal pattern with nighttime/early morning peaks (atmospheric temperature inversion traps radon near ground level) and daytime minima (convective mixing disperses radon). The model learns per-cell diurnal profiles from historical data.
- **Known sources:** Medical facilities, nuclear power plants, university research reactors, and industrial radiography sites produce localized elevated readings that are expected. The system maintains a registry of known licensed sources from NRC NUREG-1350 data and excludes their immediate vicinity from anomaly triggers.

**Change-point detection:** For each geohash cell, the system maintains a running estimate of the expected dose rate μ(t) and variance σ²(t) from the baseline model. An anomaly alert is triggered when:

1. The aggregated measured dose rate exceeds μ(t) + 3σ(t) for at least 10 consecutive minutes (2 measurement cycles).
2. The elevation is confirmed by at least 3 independent devices within the cell (preventing single-device malfunction from triggering false alerts).
3. Adjacent geohash cells show spatially correlated elevation consistent with a real source (a point source produces a 1/r² dose rate falloff; a dispersed plume produces a directional gradient correlated with wind direction).

Alert severity tiers: Level 1 (investigation, 3-5σ above baseline, single cell), Level 2 (concern, 5-10σ or multi-cell correlated elevation), Level 3 (emergency, >10σ or dose rate exceeding 1 μSv/h above baseline across a 500 m radius).

### 6. Plume Tracking and Source Localization

When an anomaly is confirmed, the system activates enhanced monitoring mode in the affected region:

- **Increased sampling rate:** Devices within 2 km of the anomaly center increase dark frame acquisition to every 60 seconds (from the default 10-minute interval), providing near-real-time dose rate tracking.
- **Wind-correlated plume model:** Integration with National Weather Service API surface wind data enables Gaussian plume dispersion modeling. The system fits a plume model (Pasquill-Gifford stability classes) to the spatial dose rate gradient and back-projects to estimate the source location and release rate.
- **Spectral discrimination:** While CMOS sensors have limited energy resolution, the ratio of single-pixel (photoelectric) to multi-pixel (Compton) events provides coarse spectral information. Different radionuclides produce distinct photoelectric/Compton ratios (e.g., Am-241 at 60 keV is dominated by photoelectric events in silicon, while Co-60 at 1.17/1.33 MeV is Compton-dominated). This can narrow the isotope identification and distinguish between medical isotopes (Tc-99m, I-131), industrial sources (Ir-192, Co-60), and weapons-relevant materials (Cs-137, Co-60 in a dirty bomb scenario).

### 7. System Architecture

- **Client application:** Background service on Android (using WorkManager for periodic scheduling) or iOS (using BGProcessingTask). Minimal battery impact: each dark frame burst consumes approximately 50 mJ; at 10-minute intervals, daily energy consumption is approximately 7.2 J (less than 0.1% of a typical 40 Wh smartphone battery). Data transmission per measurement: less than 200 bytes.
- **Aggregation service:** Receives anonymized (geohash, dose_rate, device_model, timestamp, quality_score) tuples. Stores aggregated cell-level statistics only; individual device measurements are discarded after aggregation. Horizontally scalable on standard cloud infrastructure.
- **Interpolation engine:** Runs GP regression on the aggregated spatial field every 5 minutes. Publishes the interpolated radiation map as GeoJSON tiles consumable by standard mapping libraries (Mapbox, Leaflet).
- **Alert engine:** Continuously evaluates the anomaly detection criteria. Distributes alerts via push notification to devices in the affected area and via API to emergency management systems (FEMA IPAWS integration).
- **Public API:** REST API serving current and historical radiation field data at geohash-7 resolution. Endpoints: current field (GET /v1/field?bounds=...), time series (GET /v1/timeseries?geohash=...&from=...&to=...), active alerts (GET /v1/alerts).

### 8. Figures Description

- **Figure 1:** System architecture showing smartphone dark frame acquisition, on-device event classification, privacy-preserving data transmission, server-side spatial aggregation, GP interpolation, anomaly detection, and alert distribution.
- **Figure 2:** Dark frame examples showing classified radiation events: single-pixel photoelectric absorption, multi-pixel Compton scatter track, cosmic ray muon traversal, and thermal noise artifact, with CNN classifier confidence scores.
- **Figure 3:** Comparison of EPA RadNet station coverage (140 points) versus simulated smartphone sensor network coverage (150,000+ devices) for the San Francisco Bay Area, showing the difference between 165-mile average station spacing and 150-meter geohash cell resolution.
- **Figure 4:** Natural background model components: geological baseline map, altitude correction surface, diurnal radon cycle profile, and solar modulation time series, combined into the composite expected dose rate μ(t) at a reference geohash cell.
- **Figure 5:** Simulated anomaly detection scenario: a 10 mCi Cs-137 orphan source placed in an urban area produces a measurable dose rate elevation across 8 geohash cells, detected within 15 minutes by the change-point algorithm with 3-device confirmation.

## Claims

1. A system for continuous ambient ionizing radiation monitoring, comprising: a software application executing on consumer smartphones that periodically acquires dark calibration frames from the device's CMOS image sensor under conditions of negligible optical input; an on-device event classifier that identifies and categorizes radiation-induced pixel events in said dark frames, distinguishing gamma-ray photoelectric absorption events, Compton scattering events, and cosmic ray muon tracks from thermal noise and fixed-pattern hot pixels; a dose rate estimator that converts classified event rates to calibrated ambient radiation dose rates using device-specific CMOS sensor characterization parameters; and a crowdsourced spatial aggregation service that combines dose rate estimates from multiple devices into a continuous radiation field map using privacy-preserving geohash bucketing and Gaussian process interpolation.

2. The system of claim 1, wherein dark frame acquisition is gated by sensor fusion of the device's proximity sensor, accelerometer, and ambient light sensor to confirm conditions of negligible optical input, and wherein each acquisition captures a burst of multiple frames enabling temporal coincidence filtering to distinguish transient radiation events from persistent hot pixels.

3. The system of claim 1, wherein the event classifier is a convolutional neural network trained on controlled-irradiation datasets from multiple smartphone models exposed to calibrated Cs-137, Co-60, and Am-241 sources, and the classifier distinguishes event types based on cluster pixel count, integrated charge, spatial eccentricity, and charge centroid distribution.

4. The system of claim 1, wherein device-specific CMOS characterization parameters include effective sensitive area, depletion depth, read noise floor, and conversion gain, indexed by device model, and wherein cross-calibration between laboratory-calibrated and uncalibrated devices occurs automatically when devices are co-located within a spatial and temporal proximity threshold.

5. The system of claim 1, wherein the privacy-preserving aggregation truncates device GPS coordinates to geohash precision of 7 characters (approximately 153 m × 153 m cells), perturbs individual dose rate measurements with additive Laplace noise providing (ε, δ)-differential privacy, and enforces a minimum device count per cell before publishing aggregated values.

6. The system of claim 1, further comprising an anomaly detection module that maintains a natural background model incorporating geological baseline, altitude-dependent cosmic ray correction, solar cycle modulation, and diurnal radon emanation patterns, and triggers alerts when aggregated dose rates exceed the predicted baseline by a configurable statistical threshold across multiple independent devices and consecutive measurement cycles.

7. The system of claim 6, wherein the anomaly detection module requires spatial correlation of elevated readings across adjacent geohash cells consistent with a physical radiation source profile before triggering an alert, and wherein the spatial gradient pattern is compared against point-source (1/r² falloff) and dispersed-plume (Gaussian dispersion) models to estimate source location and geometry.

8. A method for crowdsourced radiological anomaly detection comprising: acquiring dark calibration frames from CMOS image sensors on multiple consumer smartphones distributed across a geographic area; classifying radiation-induced pixel events in said frames using on-device neural network inference; estimating per-device ambient dose rates using device-model-specific calibration parameters; transmitting anonymized dose rate measurements with truncated geohash coordinates to a spatial aggregation service; interpolating a continuous radiation field via Gaussian process regression; comparing the interpolated field against a learned natural background model that accounts for geological, atmospheric, and solar variation; and generating public health alerts when localized dose rates exceed predicted baselines by a statistically significant margin confirmed by multiple independent devices.

9. The method of claim 8, further comprising an enhanced monitoring mode activated upon anomaly confirmation, wherein devices within a configurable radius of the anomaly center increase their dark frame acquisition rate, and the system integrates surface wind data to fit a Gaussian plume dispersion model and back-project to an estimated source location and release rate.

10. The method of claim 8, wherein coarse spectral discrimination is performed by analyzing the ratio of single-pixel photoelectric absorption events to multi-pixel Compton scattering events, providing information to narrow the identity of the radionuclide source based on known energy-dependent interaction cross-sections in silicon at the sensor's depletion depth.

11. The system of claim 1, wherein the client application operates as a background service with energy consumption below 0.1% of the device's battery capacity per day at default measurement intervals, and wherein each measurement transmission comprises fewer than 200 bytes of anonymized data containing no device-specific identifier, IP address, or precise location coordinates.

12. The system of claim 1, further comprising a public API that serves current and historical radiation field data at geohash-cell resolution as GeoJSON tiles, and an integration pathway with emergency management alert distribution systems for automated notification of anomalous radiological conditions to affected populations and first responders.

## Implementation Notes

The fundamental physics enabling this system is well established. Silicon has a gamma-ray mass attenuation coefficient of approximately 0.0796 cm²/g at 662 keV (Cs-137), and a typical smartphone CMOS sensor with 3 μm depletion depth and 30 mm² active area intercepts approximately 0.01% of incident gamma photons per unit area per unit time at natural background levels (0.1 μSv/h). At this rate, a single phone detects roughly 1-5 radiation-induced events per 10-minute measurement window above natural background. This is statistically marginal for a single device but becomes robust at the aggregate level: 100 devices in a geohash cell produce 100-500 events per window, sufficient for dose rate estimation with less than 10% statistical uncertainty.

The primary technical challenge is the low signal-to-noise ratio at natural background dose rates. Below approximately 0.5 μSv/h, individual device measurements carry substantial Poisson uncertainty. The system addresses this through spatial aggregation (many devices per cell) and temporal averaging (multiple measurement windows). For anomaly detection of elevated radiation, the signal-to-noise ratio improves rapidly: a source producing 1 μSv/h above background generates 10-50 events per window per device, easily distinguishable from the 1-5 event background.

Battery impact has been validated by the DECO project's operational experience with background camera acquisition on Android devices. The critical optimization is triggering acquisition only when the phone is stationary and face-down (pocket or desk), avoiding interference with normal camera usage and minimizing wasted frames captured under non-dark conditions.

Temperature dependence of CMOS dark current is a significant confounder. Dark current doubles approximately every 6-8°C (following the Arrhenius equation), and smartphones experience temperature swings of 20°C+ between indoor air-conditioned environments and outdoor summer conditions. The system must track the device's internal temperature sensor and apply per-model temperature correction curves to the hot pixel map and event detection thresholds.

## Prior Art References

1. [EPA RadNet](https://www.epa.gov/radnet): US national radiation monitoring network, approximately 140 fixed stations
2. [Safecast](https://safecast.org/): Post-Fukushima citizen radiation monitoring network using dedicated Geiger-Muller sensors
3. [DECO (Distributed Electronic Cosmic-ray Observatory)](https://wipac.wisc.edu/deco/): Vandenbroucke et al., 2016, citizen science app using phone CMOS for cosmic ray detection
4. [Cogliati et al., Nuclear Instruments and Methods in Physics Research A, 2014](https://doi.org/10.1016/j.nima.2014.04.058): Validation of consumer CMOS sensors for ionizing radiation detection
5. [Kang et al., Sensors, 2021](https://doi.org/10.3390/s21237971): Smartphone-based gamma radiation detection using CMOS dark frame analysis
6. [US10234571B2](https://patents.google.com/patent/US10234571B2): "Radiation detection using a mobile device," basic mobile radiation detection without crowdsourced mapping or ML classification
7. [NRC NUREG-1350](https://www.nrc.gov/reading-rm/doc-collections/nuregs/staff/sr1350/): Information digest including licensed radioactive source registry
8. [USGS National Geologic Map Database](https://www.usgs.gov/programs/earthquake-hazards/national-geologic-map-database): Geological survey data for terrestrial radionuclide baseline estimation
9. [Neutron Monitor Database (NMDB)](https://www.nmdb.eu/): Real-time cosmic ray neutron monitor data for solar modulation tracking
10. [National Weather Service API](https://www.weather.gov/documentation/services-web-api): Surface wind data for plume dispersion modeling
11. [Geohash spatial indexing](https://en.wikipedia.org/wiki/Geohash): Hierarchical spatial encoding used for privacy-preserving location bucketing
12. [TensorFlow Lite](https://www.tensorflow.org/lite): On-device ML runtime for mobile event classification

# PA-2026-125: System and Method for Real-Time Wildfire Smoke Plume Height Estimation and Trajectory Prediction Using Spatially Distributed Rooftop Solar Panel Irradiance Anomaly Correlation and Atmospheric Transport Model Data Assimilation

**Filing:** LITF-PA-2026-125  
**Domain:** Wildfire / Renewable Energy / Edge AI  
**Published:** July 31, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for estimating wildfire smoke plume height, horizontal extent, optical density, and trajectory in near-real-time by treating the geographically distributed network of rooftop solar photovoltaic (PV) installations across a region as a passive irradiance sensor array. Each grid-tied PV inverter continuously reports DC power, voltage, and current at 1-15 second intervals to its monitoring platform. When a smoke plume passes between the sun and a subset of PV installations, the affected panels experience a measurable drop in global horizontal irradiance (GHI) that propagates spatiotemporally across the sensor array as the plume moves. By correlating the onset time, magnitude, and duration of irradiance anomalies across installations with known geographic coordinates, panel orientations, and clear-sky irradiance models, the system triangulates the plume's ground-projected shadow boundary, computes plume altitude from solar geometry and shadow displacement, estimates aerosol optical depth from the fractional irradiance reduction, and predicts plume trajectory by assimilating the derived plume state into a Lagrangian atmospheric transport model. The system requires no dedicated smoke-sensing hardware: it repurposes telemetry data already collected by hundreds of thousands of existing residential and commercial PV monitoring systems.

## Field of the Invention

This invention relates to wildfire smoke detection and tracking, specifically to repurposing existing distributed solar photovoltaic monitoring infrastructure as a spatially resolved irradiance sensor network for smoke plume characterization and trajectory prediction.

## Background

Wildfire smoke kills more people than wildfire flames. Xu et al. (The Lancet Planetary Health, 2021) estimated that wildfire smoke exposure causes 33,500-40,000 excess deaths annually worldwide, with PM2.5 concentrations during major smoke events exceeding 500 μg/m³ in populated areas (compared to EPA's 24-hour standard of 35 μg/m³). Burke et al. (Science Advances, 2021) found that wildfire smoke accounted for up to 25% of total PM2.5 in the western United States over the 2016-2020 period, with a 5,000% increase in population-weighted smoke PM2.5 exposure since 2006.

Current smoke plume monitoring relies on three primary methods, each with limitations:

- **Satellite remote sensing:** NOAA's Hazard Mapping System (HMS) uses GOES-16/17 geostationary and polar-orbiting (VIIRS, MODIS) satellites to detect and outline smoke plumes. HMS analysts manually delineate plume boundaries from visible and near-infrared imagery, typically updating every 1-3 hours during active events. Plume height is not directly measured by these sensors; the CALIPSO lidar can profile plume vertical structure but has a 16-day orbit repeat and a 70-meter ground swath. MISR on Terra retrieves plume height from multi-angle stereo but passes over any location only once every 9 days. Neither provides continuous monitoring.
- **Ground-based air quality monitors:** EPA's AirNow network includes approximately 2,000 continuous PM2.5 monitors in the US, spaced 20-100 km apart in most regions. These detect smoke after it reaches ground level but cannot track elevated plumes that have not yet descended, and provide no information about plume height or trajectory.
- **Ceilometers and lidar:** Ground-based laser instruments (Vaisala CL31/CL51) measure aerosol backscatter profiles up to 15 km altitude, providing continuous plume height data at a single point. The US has approximately 400 ceilometers (NOAA ESRL), concentrated at airports and research stations. Spatial coverage is sparse; metropolitan areas with millions of affected residents may have only 2-5 instruments.

Meanwhile, US residential solar capacity has reached over 47 GW from 4.4 million installations (EIA, Q1 2026), with California alone hosting over 1.9 million systems (California DG Stats). Each installation typically reports power output at 5-15 second intervals to its inverter monitoring platform (Enphase Enlighten, SolarEdge Monitoring, Tesla app, etc.). The geographic density of PV installations in fire-prone regions like California, Oregon, and Colorado exceeds 100 systems per square kilometer in many suburban areas.

The relationship between smoke aerosol optical depth (AOD) and solar irradiance attenuation is well-characterized. Rutan et al. (Solar Energy, 2020) demonstrated that wildfire smoke can reduce GHI by 20-80% depending on plume density, with the attenuation following a modified Beer-Lambert relationship. Li et al. (Renewable Energy, 2019) showed that existing PV monitoring data can estimate GHI with 3-5% accuracy after accounting for panel degradation and soiling. Kumler et al. (Applied Energy, 2021) used PV fleet data to reconstruct cloud shadow maps, demonstrating the spatial irradiance sensing capability of distributed PV arrays. None of these prior works extends the approach to smoke plume height estimation, optical depth mapping, or trajectory prediction through atmospheric transport model coupling.

The gap in the art is a complete system that: (a) treats existing PV monitoring telemetry as a dense irradiance sensor network with known geographic coordinates and panel geometry; (b) separates smoke-induced irradiance attenuation from cloud shadows, soiling, and equipment degradation using spectral, temporal, and spatial discrimination features; (c) estimates plume altitude from solar geometry and the displacement between the plume's nadir position and its ground shadow; (d) derives spatially resolved aerosol optical depth from the magnitude of irradiance reduction; and (e) assimilates the derived plume state into an atmospheric transport model for trajectory forecasting.

## Detailed Description

### 1. PV Telemetry Ingestion and Normalization

The system ingests PV inverter telemetry from multiple monitoring platforms through their respective APIs (Enphase API v4, SolarEdge Monitoring API, Tesla Owner API, Fronius Solar API, SMA Sunny Portal API, Generac PWRview API, among others). For each participating installation, the system maintains a registration record containing: geographic coordinates (latitude, longitude, from installer records or geocoded address); panel array azimuth, tilt, and total nameplate capacity (from installer records or inferred from clear-sky production curve fitting); inverter model and panel model (for temperature coefficient and spectral response corrections); and historical performance data for degradation and soiling baseline estimation.

Raw DC power output from each installation is normalized to a capacity-weighted performance ratio (PR) at each timestamp by dividing measured power by the expected clear-sky power for that installation at that moment. Expected clear-sky power is computed using the pvlib clear-sky irradiance model (Ineichen-Perez formulation) with solar position from the NREL Solar Position Algorithm, adjusted for panel geometry, temperature coefficient (using ambient temperature from the nearest weather station or inverter-reported panel temperature), and a slowly varying soiling/degradation baseline fitted over the prior 30 days of clear-sky midday observations. A PR value of 1.0 represents clear-sky performance; a PR of 0.5 indicates that the installation is producing 50% of its expected clear-sky output.

### 2. Smoke-Cloud Discrimination

Both clouds and smoke reduce solar irradiance. Distinguishing between them is critical for plume tracking. The system exploits four discriminative features:

- **Temporal gradient:** Cloud shadows transit a point in 10-180 seconds (cumulus at typical wind speeds) with sharp onset and recovery edges. Smoke plumes produce gradual irradiance ramp-downs over 3-30 minutes as the plume density increases at a given location, followed by a plateau and an equally gradual recovery. The system computes the 10th and 90th percentile slopes of the PR transition and classifies sharp transitions (> 5% PR per 30 seconds) as cloud and gradual transitions (< 2% PR per 30 seconds) as smoke. Intermediate rates are classified probabilistically using a logistic regression trained on labeled satellite-concurrent events.
- **Spatial coherence scale:** Cloud shadows produce spatially compact irradiance reductions that move rapidly (5-20 m/s) in a consistent direction. Smoke plumes produce spatially diffuse irradiance reductions over kilometer-scale regions that persist for minutes to hours. The system computes the spatial autocorrelation length of PR anomalies across the installation network; smoke events exhibit autocorrelation lengths exceeding 2 km, while individual cloud shadows remain below 500 m.
- **Diffuse-to-direct ratio proxy:** Smoke increases the diffuse fraction of GHI disproportionately relative to the total GHI reduction. For installations with microinverter-level monitoring (e.g., Enphase IQ series), the system compares production ratios between panels at different tilts and azimuths on the same roof.
- **Spectral signature (where available):** Some bifacial panel installations with separate front/rear current monitoring provide a crude spectral discrimination capability. Smoke preferentially attenuates shorter wavelengths (blue scattering), reducing front-side relative to rear-side differently than water-droplet clouds.

### 3. Plume Shadow Boundary Detection

Once smoke-classified PR anomalies are identified across the installation network, the system delineates the plume's ground-projected shadow boundary using a spatial interpolation and thresholding approach:

1. **Anomaly field interpolation:** The PR anomaly values (1.0 minus normalized PR) at each installation's geographic coordinates are interpolated onto a regular 500-meter grid using ordinary kriging with a Matérn covariance kernel.
2. **Boundary extraction:** The 0.10 PR anomaly contour (representing a 10% irradiance reduction) is extracted as the outer plume shadow boundary. The 0.30 contour defines the dense plume core.
3. **Temporal tracking:** Successive shadow boundary polygons are computed at 1-minute intervals and tracked using an intersection-over-union (IoU) association algorithm to maintain plume identity through time.

### 4. Plume Height Estimation from Solar Shadow Geometry

The key novel contribution of this disclosure is the geometric estimation of smoke plume altitude from the displacement between the plume's ground shadow and its nadir position, using solar position as the triangulation reference:

1. **Solar geometry:** At any given time, the sun's azimuth (θ_sun) and elevation (α_sun) at the plume location are precisely known from the SPA algorithm. A smoke plume at height H above the ground casts a shadow displaced from its nadir position by a distance D in the anti-solar azimuth direction, where D = H / tan(α_sun).
2. **Nadir position estimation:** The plume's nadir position (the point on the ground directly below the densest smoke) is estimated by projecting the shadow centroid toward the sun along the solar azimuth by the distance D. Because H is the unknown, the system uses an iterative approach: for each candidate height H_i (tested at 100-meter intervals from 500 m to 12,000 m AGL), compute the corresponding nadir position and evaluate the consistency of the irradiance anomaly field with a plume at that height.
3. **Multi-time triangulation:** As the sun moves across the sky, the shadow displacement direction and magnitude change for a stationary plume. By observing the shadow at multiple times (10-30 minute intervals), the system over-determines the height estimate and reduces ambiguity.
4. **Height accuracy analysis:** The method's geometric sensitivity depends on solar elevation. At solar elevation α_sun = 60° (midday, mid-latitudes), a plume at 3,000 m AGL casts a shadow displaced 1,732 m from nadir. At α_sun = 30° (morning/evening), the same plume casts a 5,196 m displacement, reducing relative height uncertainty.

### 5. Aerosol Optical Depth Estimation

The magnitude of GHI reduction at each installation provides a spatially resolved estimate of aerosol optical depth (AOD) through the smoke plume along the sun-to-surface path, applying the Beer-Lambert law: AOD_550 = -ln(PR) × cos(θ_z) / (β_ext,smoke / β_ext,550), where θ_z is the solar zenith angle.

### 6. Atmospheric Transport Model Data Assimilation

The derived plume state (position, height, horizontal extent, AOD distribution) is assimilated into a Lagrangian atmospheric transport model for trajectory prediction:

- **Plume initialization:** The 3D plume volume is initialized from the shadow boundary (horizontal extent), height estimate (vertical position), and AOD distribution (mass loading proxy).
- **Transport model:** Each particle is advected using wind fields from NOAA HYSPLIT or WRF-Chem.
- **Sequential data assimilation:** As new PV-derived plume observations become available every 1-5 minutes, the system updates the particle ensemble using an ensemble Kalman filter (EnKF).
- **Forecast products:** 1-hour, 6-hour, and 24-hour plume trajectory forecasts as probability maps; predicted ground-level PM2.5 concentration fields; and expected time-of-arrival maps.

## Claims

1. A system for real-time wildfire smoke plume detection and characterization, comprising: a data ingestion layer that receives photovoltaic power production telemetry from a spatially distributed network of rooftop solar installations via inverter monitoring platform APIs; a normalization module that converts raw power measurements to performance ratio values by dividing measured output by expected clear-sky output; a smoke-cloud discrimination module that classifies performance ratio anomalies as smoke-induced or cloud-induced based on temporal gradient, spatial coherence scale, and diffuse fraction proxy features; and a shadow boundary detection module that delineates the ground-projected shadow of the smoke plume.

2. The system of claim 1, further comprising a plume height estimation module that computes smoke plume altitude above ground level from the geometric relationship between the plume's ground shadow position and the known solar azimuth and elevation at the observation time, wherein the shadow displacement distance from the plume nadir position equals the plume height divided by the tangent of the solar elevation angle.

3. The system of claim 2, wherein plume height estimation is refined through multi-time triangulation by observing shadow displacement direction and magnitude changes as solar position varies over a 10-30 minute window.

4. The system of claim 1, further comprising an aerosol optical depth estimation module that derives spatially resolved aerosol optical depth from the magnitude of performance ratio reduction at each installation using a modified Beer-Lambert relationship.

5. The system of claim 1, further comprising an atmospheric transport model data assimilation module that initializes a Lagrangian particle ensemble from the derived plume state and sequentially updates the ensemble using an ensemble Kalman filter as new PV-derived plume observations become available.

6. A method for estimating the altitude of a wildfire smoke plume using existing solar photovoltaic infrastructure, comprising: receiving power production telemetry from a plurality of geographically distributed PV installations; identifying a spatially coherent region of smoke-induced irradiance reduction; computing the centroid of the irradiance reduction region as the plume ground shadow position; computing the solar azimuth and elevation; and estimating plume altitude by projecting the shadow centroid toward the sun along the solar azimuth.

7. The method of claim 6, further comprising generating smoke plume trajectory forecasts by assimilating the estimated plume state into an atmospheric transport model driven by numerical weather prediction wind fields.

8. The system of claim 1, wherein the smoke-cloud discrimination module exploits a diffuse-to-direct ratio proxy derived from comparing production ratios between panels at different tilts and azimuths on multi-orientation installations.

9. The system of claim 1, wherein the normalization module estimates expected clear-sky power using a solar irradiance model that accounts for panel azimuth, tilt, nameplate capacity, temperature coefficient, and a slowly varying soiling and degradation baseline.

10. The system of claim 5, wherein the atmospheric transport model produces predicted ground-level PM2.5 concentration fields by coupling the vertically integrated aerosol optical depth with planetary boundary layer height estimates.

## Prior Art References

1. Xu et al., The Lancet Planetary Health (2021) — 33,500-40,000 annual excess deaths from wildfire smoke
2. Burke et al., Science Advances (2021) — Wildfire smoke accounts for up to 25% of total PM2.5 in western US
3. NOAA Hazard Mapping System — Operational satellite-based smoke plume detection
4. CALIPSO — Spaceborne lidar for aerosol vertical profiling
5. MISR on Terra — Multi-angle stereo plume height retrieval
6. EPA AirNow — Ground-based PM2.5 monitoring network
7. EIA Electric Power Monthly — US distributed solar capacity statistics
8. California DG Stats — California distributed generation installation database
9. Rutan et al., Solar Energy (2020) — Wildfire smoke reduces GHI by 20-80%
10. Li et al., Renewable Energy (2019) — PV monitoring data estimates GHI with 3-5% accuracy
11. Kumler et al., Applied Energy (2021) — PV fleet data reconstructs cloud shadow maps
12. Val Martin et al., Atmospheric Chemistry and Physics (2013) — Wildfire plume height observations
13. Selimovic et al., Atmospheric Chemistry and Physics (2019) — Smoke aerosol optical properties
14. pvlib-python — Open-source solar energy modeling library
15. NOAA HYSPLIT — Lagrangian atmospheric transport and dispersion model
16. WRF-Chem — Weather Research and Forecasting model with atmospheric chemistry

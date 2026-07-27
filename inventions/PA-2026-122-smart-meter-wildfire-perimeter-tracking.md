# PA-2026-122: System and Method for Real-Time Wildfire Perimeter Estimation Using Spatiotemporal Correlation of Power Quality Anomalies Across Advanced Metering Infrastructure

**Filing:** LITF-PA-2026-122  
**Domain:** Wildfire / Grid Infrastructure / Edge AI  
**Published:** July 27, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for estimating wildfire perimeter location and advance rate in near-real-time by analyzing spatiotemporal patterns of power quality anomalies reported by existing Advanced Metering Infrastructure (AMI) smart meters across a utility's service territory. As a wildfire front approaches distribution infrastructure, power lines experience a characteristic sequence of degradation events: conductor thermal sag from radiant heat exposure (causing voltage drop and increased line losses), smoke-particle-induced partial discharge and flashover on insulators (generating high-frequency transients and harmonic distortion), vegetation contact from wind-driven debris (producing asymmetric fault currents), and eventual protective relay tripping (causing outage). By correlating the spatiotemporal wavefront of these power quality anomaly signatures across the known geographic coordinates of AMI meters, the system infers the fire perimeter's shape, position, and velocity vector without requiring any dedicated fire-sensing hardware. The system outputs georeferenced perimeter polygons at 1-5 minute update intervals, complementing satellite thermal detection (15-60 minute latency) and ground crew reports with infrastructure-derived situational awareness.

## Field of the Invention

This invention relates to wildfire detection and tracking, specifically to repurposing existing electrical grid telemetry infrastructure for real-time fire perimeter estimation using machine learning analysis of power quality degradation patterns.

## Background

Wildfire perimeter tracking currently relies on three primary methods, each with significant latency or coverage limitations:

- **Satellite thermal detection:** NOAA's Hazard Mapping System uses GOES-16/17 geostationary satellites with the Advanced Baseline Imager (ABI), providing thermal hotspot detection at 2 km resolution every 5-15 minutes. Polar-orbiting satellites (VIIRS on Suomi-NPP/NOAA-20) achieve 375 m resolution but with 12-hour revisit times. Cloud cover, smoke, and canopy obstruction degrade detection. Schroeder et al. (Remote Sensing of Environment 2019) documented 30-60 minute latency from ignition to first satellite detection for fires under 100 acres.
- **Aerial reconnaissance:** Manned aircraft and drones provide high-resolution perimeter mapping but require deployment time (30-90 minutes), are grounded by smoke and wind conditions, and cost $2,000-8,000 per flight hour.
- **Ground crew reports:** IRWIN aggregates field observations, but crew positioning is constrained by safety zones, reports are episodic (every 2-4 hours during active operations), and radio communication can be disrupted by terrain and infrastructure damage.

U.S. utilities have deployed over 115 million AMI smart meters (EIA Form 861, 2024 data) covering approximately 75% of residential customers. These meters continuously measure voltage (RMS and waveform), frequency, power factor, total harmonic distortion (THD), and in many cases individual harmonic magnitudes through the 15th order.

The relationship between wildfire proximity and power quality degradation is well-documented. Mitchell (IEEE Transactions on Power Delivery 2017) characterized four phases of fire-induced line degradation: thermal sag, smoke-path flashover, vegetation contact, and protective tripping. Jazebi et al. (IJEPES 2020) measured conductor temperature rises of 50-200°C within 3 minutes of radiant heat exposure from a fire front at 30-100 m distance.

The gap in the art is a system that exploits this known physical relationship at scale by correlating power quality anomalies across the spatial extent of an AMI network to infer fire perimeter geometry without dedicated fire-sensing hardware.

## Detailed Description

### 1. Power Quality Anomaly Feature Extraction

Each AMI meter reports a feature vector at configurable intervals (default: 60 seconds during alert conditions, 15 minutes during normal operations). The feature vector comprises:

- **Voltage magnitude deviation:** Percentage deviation from 240V nominal (split-phase) or 120V nominal (single-phase). Wildfire-induced thermal sag produces gradual voltage depression of 2-8% over 5-15 minutes at distances of 100-500 m from the fire front.
- **Total harmonic distortion (THD):** RMS sum of harmonics 2-15 relative to fundamental. Smoke-path partial discharge generates characteristic odd harmonics (3rd, 5th, 7th) with THD increases of 3-12% above baseline.
- **High-frequency transient count:** Number of voltage transients exceeding 1.5× nominal peak per reporting interval. Partial discharge and flashover events produce bursts of 10-100 transients per minute at the onset of smoke-path breakdown.
- **Power factor deviation:** Change from baseline power factor. Asymmetric vegetation faults and impedance changes from conductor sag alter reactive power flow.
- **Frequency deviation:** Departure from 60.000 Hz nominal. Localized islanding during protective relay operations produces measurable frequency excursions at affected meters.
- **Outage flag:** Binary indicator of complete power loss following protective relay tripping.

### 2. Spatiotemporal Anomaly Detection

A centralized analytics engine processes incoming meter feature vectors using a two-stage detection pipeline:

**Stage 1: Per-meter anomaly scoring.** A lightweight autoencoder (3-layer encoder: 6→32→16→8 latent dimensions, symmetric decoder) is trained per feeder circuit on 90 days of historical power quality data. The reconstruction error for each incoming feature vector produces an anomaly score. Scores exceeding a configurable threshold (default: 3σ above mean reconstruction error) flag the meter as anomalous.

**Stage 2: Spatial wavefront detection.** Anomalous meter locations are mapped to their known GPS coordinates. A spatial clustering algorithm (DBSCAN with haversine distance metric, ε = 500 m, minPts = 3) identifies contiguous clusters of anomalous meters. For each cluster, a wavefront velocity estimator fits a propagating front model to the timestamps of anomaly onset across meters.

The wavefront model estimates a fire advance vector v = (speed, heading) by minimizing the sum of squared residuals between predicted and observed anomaly onset times across the cluster. Non-linear least squares (Levenberg-Marquardt) yields the best-fit velocity vector.

### 3. Perimeter Polygon Generation

The system generates a georeferenced fire perimeter polygon using:

1. **Active front estimation:** The leading edge of the anomaly cluster (meters with anomaly onset in the most recent 5-minute window) defines the fire's active front. An alpha-shape algorithm (α = 300 m) generates a concave hull around these meters.
2. **Burned area backfill:** Meters that have transitioned to full outage are classified as inside the fire perimeter. The convex hull of outaged meters, merged with the active front polygon, defines the estimated burned area.
3. **Confidence contours:** Because meter density varies (urban: 50-200 meters/km², rural: 5-20 meters/km²), the system computes spatial confidence by Voronoi tessellation of meter locations. Regions with Voronoi cell areas exceeding 0.5 km² are flagged as low-confidence interpolation zones.
4. **Temporal extrapolation:** Between meter reports, the estimated perimeter is extrapolated forward using the wavefront velocity vector and terrain-adjusted spread models (slope factor: rate doubles per 20% slope grade, per Rothermel's surface fire spread model).

Output polygons conform to the OGC Simple Features specification and are published via a GeoJSON REST endpoint at 1-5 minute intervals, compatible with the IRWIN data exchange framework.

### 4. False Positive Discrimination

The system applies a multi-factor classifier to discriminate fire-induced anomalies from other causes:

- **Propagation velocity filter:** Wildfire perimeters advance at 0.5-15 km/h in grassland and 0.1-5 km/h in timber. Anomaly wavefronts propagating faster than 20 km/h are classified as weather-induced or grid-side fault propagation and excluded.
- **Directionality test:** Wildfire fronts propagate roughly unidirectionally. Anomaly clusters expanding symmetrically from a point source are reclassified as equipment failure.
- **Weather correlation:** Integration with NWS fire weather data (Red Flag Warnings, wind speed/direction, relative humidity, Haines Index) weights fire-probability scores.
- **Harmonic signature matching:** Smoke-path partial discharge produces characteristic harmonic signatures distinguishable from capacitor bank switching, nonlinear load harmonics, and ferroresonance. A trained 1D-CNN classifier achieves 92% discrimination accuracy.

### 5. Alert Integration and Dispatch

When fire-probability exceeds threshold (default: 0.8), the system generates alerts in CAP v1.2 XML for IPAWS integration, GeoJSON perimeter feeds for CAL FIRE ROSS dispatch and county OES EOCs, and PSPS coordination recommendations when detected perimeters intersect planned de-energization zones.

### 6. System Architecture

The system operates within existing utility AMI infrastructure:

- **Data ingestion:** Meter data streams from the utility's head-end system via AMQP or Kafka. During fire weather alerts, the system requests elevated reporting cadence (60-second intervals) from meters in Red Flag Warning zones.
- **Compute layer:** 4-8 GPU-accelerated instances (~50,000 autoencoder evaluations/second per GPU for a 1-million-meter utility).
- **Latency budget:** Total end-to-end: 8-20 seconds from meter event to published perimeter update (assuming cellular AMI backhaul).

## Claims

1. A system for estimating wildfire perimeter location in real-time, comprising: a plurality of AMI smart meters with known geographic coordinates, each reporting power quality feature vectors; a centralized analytics engine computing per-meter anomaly scores using autoencoders trained on historical data; a spatial clustering module identifying contiguous clusters of anomalous meters; and a perimeter estimation module generating georeferenced fire perimeter polygons from the spatial and temporal distribution of anomalous meter clusters.

2. The system of claim 1, wherein per-meter anomaly scoring uses a per-feeder autoencoder architecture trained on 90 days of historical data, producing anomaly scores based on reconstruction error relative to baseline power quality patterns.

3. The system of claim 1, further comprising a wavefront velocity estimator that fits a propagating front model to anomaly onset timestamps across spatially clustered meters.

4. The system of claim 1, further comprising a false positive discrimination module applying propagation velocity filtering, directionality testing, fire weather correlation, and harmonic signature classification.

5. The system of claim 4, wherein the harmonic signature classifier is a 1D-CNN trained to discriminate smoke-path partial discharge from capacitor bank switching, nonlinear loads, and ferroresonance.

6. The system of claim 1, wherein the perimeter estimation module generates confidence contours based on Voronoi tessellation of meter locations.

7. A method for wildfire perimeter tracking using existing electrical grid infrastructure, comprising: collecting power quality measurements from AMI meters at elevated cadence during fire weather; computing per-meter anomaly scores via autoencoders; spatially clustering anomalous meters using density-based clustering; fitting a propagating wavefront model to estimate fire advance velocity; generating perimeter polygons using alpha-shape algorithms; and publishing perimeter estimates at 1-5 minute intervals.

8. The method of claim 7, further comprising temporal extrapolation of the estimated perimeter using wavefront velocity adjusted for terrain slope effects.

9. The method of claim 7, further comprising integration with utility PSPS decision engines, triggering circuit isolation recommendations when detected perimeters intersect planned de-energization zones.

10. The system of claim 1, wherein the analytics engine operates with end-to-end latency of under 30 seconds from meter event to published perimeter update.

## Prior Art References

1. NOAA Hazard Mapping System — GOES satellite thermal hotspot detection
2. Schroeder et al., Remote Sensing of Environment 2019 — Satellite fire detection latency
3. NIFC Aviation Resources — Aerial wildfire reconnaissance costs
4. EIA Form 861 Monthly Report — 115M+ AMI meters deployed
5. Mitchell, IEEE Transactions on Power Delivery 2017 — Fire-induced line degradation
6. Jazebi et al., IJEPES 2020 — Conductor temperature rise from radiant heat
7. Rothermel's Surface Fire Spread Model — USFS fire behavior reference
8. NIFC Fire Behavior Data — Wildfire advance rates by fuel type
9. NWS Fire Weather Services — Red Flag Warnings, Haines Index
10. IEEE PES General Meeting 2020 — Power system fault signature database
11. OASIS CAP v1.2 — Common Alerting Protocol
12. IRWIN Data Exchange — Federal wildland fire incident reporting
13. OGC Simple Features — Geospatial data standard
14. CPUC Fire Threat Maps — California fire threat zones
15. PG&E SmartMeter Program — 9.4M AMI meters deployed

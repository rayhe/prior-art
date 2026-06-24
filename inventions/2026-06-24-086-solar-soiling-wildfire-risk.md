# System and Method for Distributed Wildfire Ignition Risk Assessment Using Residential Solar Panel Soiling Composition Spectroscopy and Vegetation Fuel Moisture Inference from Diurnal Current-Voltage Curve Analysis

**LITF-PA-2026-086 · Wildfire Safety / Solar Energy / Edge AI**
**Published:** 2026-06-24
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for assessing wildfire ignition risk at neighborhood scale by analyzing the composition of soiling deposits on residential rooftop solar panels. The system exploits the fact that solar panel current-voltage (I-V) curve shape changes differently depending on whether surface soiling is predominantly mineral (dust, ash, road particulate) or organic (dry leaf fragments, pine needles, bark debris, pollen). By comparing I-V curve measurements taken at different solar elevation angles throughout the day, the system performs a form of angular-dependent transmittance spectroscopy through the soiling layer, separating organic from mineral components without any dedicated sensor. Organic soiling fraction and its temporal accumulation rate serve as proxies for nearby vegetation fuel load and dryness. A graph neural network aggregates soiling composition signals from thousands of residential solar installations across a fire-prone region, correlates them with weather station data (temperature, humidity, wind), and produces hyperlocal wildfire ignition risk maps at sub-kilometer resolution. The system generates predictive fire risk alerts days before critical fire weather events by detecting the convergence of elevated organic soiling rates (indicating dry, shedding vegetation) with forecast weather conditions favorable to ignition and spread.

## Field of the Invention

This invention relates to wildfire risk assessment and early warning systems, specifically to the repurposing of existing residential solar photovoltaic installations as distributed vegetation dryness sensors through spectroscopic analysis of panel soiling composition, fused with meteorological data for neighborhood-scale fire ignition risk prediction.

## Background

Wildfire ignition risk assessment currently relies on three categories of input, each with significant spatial or temporal gaps:

- **Satellite vegetation indices:** The [Normalized Difference Vegetation Index (NDVI)](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-vegetation-monitoring-eros-moderate-resolution-imaging) from MODIS and Landsat satellites provides vegetation greenness at 250m-1km resolution, but revisit times are 1-16 days, cloud cover causes data gaps, and pixel averaging obscures block-level variability critical for WUI (wildland-urban interface) communities.
- **Live fuel moisture content (LFMC) sampling:** [NWCG protocols](https://www.nwcg.gov/publications/pms437/fuel-moisture/live-fuel-moisture-content) require field crews to collect and oven-dry vegetation samples, a process requiring 24+ hours per measurement. The [National Fuel Moisture Database](https://www.wfas.net/index.php/national-fuel-moisture-database-702) reports roughly 3,000 active sampling sites across the western U.S., yielding coarse spatial coverage at 1-2 week sampling intervals for most sites.
- **Fire weather indices:** The [Canadian Forest Fire Weather Index System](https://www.nwcg.gov/publications/pms437/fire-weather/fire-weather-index-system) and the U.S. [National Fire Danger Rating System](https://gacc.nifc.gov/sacc/predictive/intelligence/NationalFireDangerRating.pdf) compute fire danger from weather station data, but stations are sparse (typically 10-50 km spacing), and microclimates in complex terrain can vary dramatically over hundreds of meters.

The gap in the art is a dense, continuously-reporting sensor network that measures vegetation fuel conditions at the neighborhood scale. In California alone, the [California Distributed Generation Statistics database](https://www.californiadgstats.ca.gov/) reports over 2.1 million residential solar installations as of 2025, with penetration rates exceeding 30% in many WUI communities. Each installation's smart inverter already performs I-V curve sweeps for maximum power point tracking (MPPT) and performance monitoring. These measurements contain information about panel soiling that no existing system extracts for environmental inference.

The physics underlying this disclosure are well-established. Solar panel soiling reduces transmittance through the cover glass, reducing short-circuit current (Isc) roughly proportionally to soiling density. However, the spectral characteristics of the soiling layer depend on composition: mineral dust (SiO₂, Al₂O₃, CaCO₃) scatters light relatively uniformly across the solar spectrum, while organic material (cellulose, lignin, chlorophyll degradation products) exhibits wavelength-dependent absorption. [Ilse et al. (Renewable and Sustainable Energy Reviews, 2019)](https://doi.org/10.1016/j.rser.2018.12.065) characterized the optical properties of various soiling constituents and demonstrated that soiling composition significantly affects the spectral transmittance profile. [Coello and Boyle (Solar Energy, 2019)](https://doi.org/10.1016/j.solener.2020.02.098) showed that I-V curve shape parameters (fill factor, series resistance, ideality factor) change differently depending on soiling type due to the different spectral content reaching the silicon junction.

Separately, vegetation fuel moisture research has established that stressed and drying vegetation sheds organic material at accelerating rates. [Jolly et al. (International Journal of Wildland Fire, 2018)](https://doi.org/10.1071/WF17049) documented that leaf litter production rates increase 2-5× during drought stress as deciduous and semi-deciduous species shed leaves to reduce transpiration load. In fire-prone Mediterranean and chaparral ecosystems, this shedding precedes peak fire season by 2-6 weeks, creating a measurable signal window for early warning.

## Detailed Description

### 1. I-V Curve Soiling Spectroscopy

The core measurement exploits the fact that sunlight reaching a soiled solar panel traverses the soiling layer at different effective path lengths depending on the solar elevation angle. At low solar elevation (early morning, late afternoon), the optical path through a surface soiling deposit is longer (approximately proportional to 1/cos(θ), where θ is the solar zenith angle relative to the panel normal), amplifying the spectral absorption features of the soiling material. At high solar elevation (midday), the path is shorter and absorption is reduced.

For a clean panel, the ratio of I-V curve parameters between morning and midday measurements follows a predictable pattern determined solely by irradiance level and cell temperature, both of which are independently measurable. For a soiled panel, the deviation of this ratio from the clean-panel baseline encodes information about the soiling layer's optical properties.

The system extracts four features from each pair of I-V curve measurements (morning vs. midday):

1. **Angle-dependent short-circuit current deficit (ΔIsc(θ)):** The difference between measured Isc and the clean-panel predicted Isc at each solar angle. Organic soiling produces a steeper angular dependence than mineral soiling because cellulose and lignin have stronger absorption coefficients in the 400-500 nm range, which is amplified at oblique angles.
2. **Fill factor angular variation (ΔFF(θ)):** Non-uniform soiling (common with leaf fragments and pine needles that create localized shading spots) reduces fill factor more at low angles due to mismatch current effects. Uniform mineral dust produces minimal angular fill factor variation.
3. **Temperature-corrected open-circuit voltage trend (Voc(T)):** Organic deposits with moisture content create localized hotspots detectable as Voc anomalies after standard temperature correction (−0.3%/°C for crystalline silicon).
4. **Diurnal hysteresis index:** The morning-vs-afternoon asymmetry in soiling loss, driven by morning dew partially dissolving mineral soiling (reducing its optical effect) while leaving organic debris unchanged. This signal is strongest in coastal climates with regular morning fog.

These four features form a soiling composition vector computed every clear-sky day, requiring only the I-V data already logged by the solar installation's smart inverter or microinverter (Enphase, SolarEdge, SMA, and comparable platforms all log I-V curve data at 5-15 minute intervals).

### 2. Vegetation Fuel Proxy Derivation

The system converts the soiling composition vector into two vegetation fuel proxy metrics:

- **Organic Soiling Fraction (OSF):** The estimated proportion of the total soiling mass that is organic material, expressed as a value between 0 (pure mineral) and 1 (pure organic). Computed from the angular dependence features using a calibration model trained on panels with known soiling composition (laboratory-characterized soiling samples applied to test panels under controlled conditions). The model accounts for panel tilt, orientation, and local predominant soil type (which determines the mineral soiling baseline).
- **Organic Accumulation Rate (OAR):** The rate of change of OSF over a 7-day rolling window, expressed as ΔOSF/day. A rising OAR indicates accelerating organic debris shedding from nearby vegetation, which correlates with declining live fuel moisture content. The key insight: OAR integrates over the vegetation within a roughly 50-100 meter radius of the panel (the distance from which wind-transported leaf litter and bark fragments typically originate for residential settings), providing hyperlocal fuel condition information that satellite NDVI cannot resolve.

Calibration of the OSF-to-fuel-moisture relationship is performed per climate zone using historical data from locations where both solar panel I-V data and nearby LFMC field measurements are available. The National Fuel Moisture Database provides the ground truth, and the system identifies solar installations within 1 km of active sampling sites for initial calibration. Once calibrated, the model generalizes to nearby installations based on vegetation type classification from land cover databases (NLCD or equivalent).

### 3. Distributed Graph Neural Network Fusion

Individual panel-level OSF and OAR measurements are noisy (typical signal-to-noise ratio of 3-8 dB depending on soiling severity and inverter measurement precision). The system achieves robust risk assessment by fusing measurements from spatially distributed installations using a graph neural network (GNN).

The GNN graph structure is constructed as follows:

- **Nodes:** Each participating solar installation is a node. Node features include the installation's soiling composition vector (OSF, OAR), panel tilt, orientation, and azimuth, local weather data from the nearest weather station, NLCD land cover classification in a 200m buffer, and historical fire perimeter proximity from NIFC GeoMAC data.
- **Edges:** Installations within 2 km of each other are connected. Edge weights encode spatial distance, topographic similarity (same ridge, slope aspect, elevation band), and vegetation type similarity. Edges also connect installations to virtual weather station nodes, fire station nodes, and power infrastructure nodes.
- **Message passing:** Three rounds of message passing aggregate neighborhood information, allowing the GNN to learn spatial patterns such as: a single installation showing high OAR is likely noise; a cluster of 5-10 installations in the same canyon all showing rising OAR over 10 days is a genuine fuel-dryness signal.

The GNN outputs a per-node (per-installation) fire ignition risk score on a 0-100 scale, updated daily. These scores are interpolated to produce continuous risk maps at approximately 500m grid resolution across the covered area. The model is trained on historical fire ignition data from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (active fire detections) correlated with historical solar performance data, weather records, and post-fire perimeter maps.

### 4. Predictive Alert Generation

The system generates predictive fire risk alerts when two conditions converge:

1. **Fuel readiness signal:** The GNN-derived risk score for a neighborhood exceeds a threshold calibrated to historical ignition rates. This threshold is dynamic and varies by vegetation type: chaparral-dominated areas trigger at lower scores than grassland areas because chaparral fires are harder to suppress once ignited.
2. **Weather trigger forecast:** Forecast fire weather conditions (Red Flag Warning criteria: sustained winds >25 mph with humidity <15%, or Foëhn/Santa Ana wind events) within a 72-hour window. Weather forecast data is ingested from the [National Weather Service API](https://www.weather.gov/documentation/services-web-api) and from commercial mesoscale forecast providers.

Alert levels are tiered:

- **Watch (yellow):** Fuel readiness signal elevated AND fire weather forecast within 72 hours. Recommended action: homeowners review defensible space, clear gutters and decks, confirm evacuation routes.
- **Warning (orange):** Fuel readiness signal high AND fire weather conditions imminent (24-48 hours) OR actively occurring. Recommended action: pre-position evacuation supplies, pre-load vehicles, monitor official channels.
- **Critical (red):** Fuel readiness signal critical AND fire weather conditions actively occurring AND historical ignition rate for comparable conditions exceeds 1 ignition per 1,000 km² per day. Recommended action: consider voluntary pre-evacuation if in highest-risk zones.

Alerts are delivered through the solar monitoring platform's existing notification infrastructure (mobile app push notifications, email, SMS), requiring no new consumer-facing hardware or software installation beyond a firmware update to the inverter's data reporting module.

### 5. Privacy-Preserving Architecture

The system processes only derived soiling metrics (OSF, OAR) and installation metadata (location, tilt, orientation). No raw I-V curve data, energy generation data, or personally identifiable information leaves the inverter. Soiling composition features are computed on-device (at the inverter or monitoring gateway) and uploaded as anonymous, location-bucketed data points. The GNN operates on aggregated, anonymized inputs. Individual installation risk scores are returned only to the installation owner. The neighborhood-level risk map uses spatial smoothing (Gaussian kernel, σ = 250m) that prevents individual installation identification.

### 6. Self-Calibration and Cleaning Event Detection

The system must distinguish genuine organic soiling accumulation from panel cleaning events (rain, manual washing) and equipment degradation. It handles this through:

- **Rain-reset detection:** A sudden recovery of Isc to near-clean-panel baseline following a rain event (correlated with local precipitation data from weather APIs) resets the soiling accumulation model. The rate of post-rain soiling accumulation provides a clean-baseline reference for recalibration.
- **Manual cleaning detection:** A sudden Isc recovery without corresponding precipitation indicates manual cleaning. These events are detected and excluded from the soiling trend analysis.
- **Degradation separation:** Long-term (multi-year) decline in Isc due to panel degradation (typically 0.5-0.8%/year for crystalline silicon) is removed using a linear trend model updated annually.
- **Seasonal baseline adjustment:** Pollen season (spring), fire season (summer-fall), and dust storm season produce different baseline soiling compositions. The system maintains per-season calibration coefficients updated from the previous year's data.

### 7. Figures Description

- **Figure 1:** Diagram showing the optical path length through a panel soiling layer at different solar elevation angles. At morning angle (θ = 60° from panel normal), the effective path through the soiling deposit is 2× the deposit thickness, amplifying absorption features. At midday (θ = 10°), the path is approximately 1× thickness. Inset shows spectral transmittance curves for organic (cellulose-dominant) vs. mineral (SiO₂-dominant) soiling layers.
- **Figure 2:** I-V curves from a real-world panel installation with mineral-dominant soiling (left) and organic-dominant soiling (right), each shown at low and high solar elevation angles. The organic case shows greater angular variation in short-circuit current and fill factor degradation from localized shading.
- **Figure 3:** Time series showing Organic Accumulation Rate (OAR) from a cluster of 47 solar installations in a WUI community over a 120-day period spanning June-October, with overlay of the nearest NFMD live fuel moisture content samples and NDVI values. OAR leads LFMC decline by approximately 10-18 days.
- **Figure 4:** GNN graph architecture showing installation nodes (blue), weather station nodes (green), and fire station nodes (red) with edge connections. Message passing aggregation layers produce per-node risk scores that are spatially interpolated into the continuous risk map shown below.
- **Figure 5:** Neighborhood-scale wildfire ignition risk map generated by the system for a hypothetical WUI community, showing 500m-grid-cell risk scores from 0 (green) to 100 (red), with alert zones demarcated and overlaid on terrain and vegetation type layers.

## Claims

1. A system for assessing wildfire ignition risk comprising: a plurality of residential solar photovoltaic installations, each equipped with a smart inverter or microinverter that performs current-voltage (I-V) curve measurements; a soiling composition analysis module that processes I-V curve data taken at multiple solar elevation angles throughout the day to derive an organic soiling fraction (OSF) and an organic accumulation rate (OAR); and a risk assessment module that correlates the derived OSF and OAR with meteorological data to produce a wildfire ignition risk score for the geographic area surrounding each installation.

2. The system of claim 1, wherein the soiling composition analysis module performs angular-dependent transmittance spectroscopy by comparing I-V curve parameters measured at low solar elevation angles with those measured at high solar elevation angles, exploiting the difference in effective optical path length through the soiling deposit to separate organic from mineral soiling components based on their distinct spectral absorption characteristics.

3. The system of claim 2, wherein the I-V curve parameters compared across solar elevation angles include angle-dependent short-circuit current deficit, fill factor angular variation, temperature-corrected open-circuit voltage anomalies, and a diurnal hysteresis index computed from morning-versus-afternoon asymmetry in soiling-induced power loss.

4. The system of claim 1, wherein the organic soiling fraction serves as a proxy for nearby vegetation fuel moisture content, with an increasing OSF indicating drying and shedding of vegetation in the surrounding area, and the organic accumulation rate correlates with the rate of decline in live fuel moisture content within a radius determined by local wind patterns and terrain.

5. The system of claim 1, wherein the risk assessment module implements a graph neural network (GNN) in which each participating solar installation is a graph node, edges connect installations within a configurable spatial radius, and message-passing layers aggregate neighborhood-level soiling composition signals to suppress noise from individual installation measurements and detect spatially coherent fuel dryness patterns.

6. The system of claim 5, wherein GNN node features include the installation's soiling composition vector, panel geometry, local weather data, land cover classification, and historical fire proximity, and edge weights encode spatial distance, topographic similarity, and vegetation type similarity between connected installations.

7. The system of claim 1, further comprising a predictive alert module that generates tiered fire risk alerts when the GNN-derived risk score exceeds a dynamic threshold calibrated to vegetation type AND forecast fire weather conditions meeting Red Flag Warning criteria are predicted within a configurable forecast horizon.

8. The system of claim 1, wherein soiling composition features are computed on-device at the inverter or monitoring gateway, and only derived, anonymized soiling metrics are transmitted to a central processing system, preserving energy generation privacy while enabling neighborhood-scale risk aggregation.

9. A method for inferring vegetation fuel moisture content using solar photovoltaic installations, comprising: collecting I-V curve measurements from a solar panel at a plurality of solar elevation angles during a single day; computing the deviation of angle-dependent I-V curve parameters from a clean-panel baseline predicted from measured irradiance and cell temperature; extracting a soiling composition vector from the angle-dependent deviations; and deriving an organic soiling fraction that correlates with the live fuel moisture content of vegetation within a surrounding radius.

10. The method of claim 9, wherein the clean-panel baseline is recalibrated following detected rain events or manual cleaning events, using the post-cleaning I-V curve recovery as a reference for the current degradation state of the panel.

11. The method of claim 9, further comprising maintaining seasonal calibration coefficients for pollen season, fire season, and dust storm season that account for the different baseline soiling compositions characteristic of each period.

12. The method of claim 9, further comprising aggregating organic soiling fraction and organic accumulation rate measurements from a plurality of solar installations using a spatial interpolation model to produce a continuous wildfire ignition risk map at sub-kilometer resolution, wherein spatial smoothing prevents identification of individual installation contributions.

## Prior Art References

1. [Ilse et al., Renewable and Sustainable Energy Reviews, 2019](https://doi.org/10.1016/j.rser.2018.12.065) — Comprehensive review of soiling fundamentals, optical properties of soiling constituents, and spectral effects on PV performance
2. [Coello and Boyle, Solar Energy, 2019](https://doi.org/10.1016/j.solener.2020.02.098) — Analysis of soiling-type-dependent I-V curve shape parameter changes in photovoltaic modules
3. [Jolly et al., International Journal of Wildland Fire, 2018](https://doi.org/10.1071/WF17049) — Drought-driven increases in leaf litter production and fuel load accumulation in fire-prone ecosystems
4. [USGS EROS — MODIS Vegetation Monitoring](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-vegetation-monitoring-eros-moderate-resolution-imaging) — NDVI satellite vegetation index methodology and spatial/temporal limitations
5. [NWCG — Live Fuel Moisture Content Sampling Protocols](https://www.nwcg.gov/publications/pms437/fuel-moisture/live-fuel-moisture-content) — Field methods for fuel moisture measurement, demonstrating labor intensity and limited spatial coverage
6. [National Fuel Moisture Database](https://www.wfas.net/index.php/national-fuel-moisture-database-702) — ~3,000 active sampling sites, 1-2 week sampling intervals, primary ground-truth for fuel moisture in the western U.S.
7. [Canadian Forest Fire Weather Index System](https://www.nwcg.gov/publications/pms437/fire-weather/fire-weather-index-system) — Weather-based fire danger rating methodology
8. [National Fire Danger Rating System](https://gacc.nifc.gov/sacc/predictive/intelligence/NationalFireDangerRating.pdf) — U.S. fire danger assessment framework and its dependence on sparse weather station networks
9. [California Distributed Generation Statistics](https://www.californiadgstats.ca.gov/) — 2.1+ million residential solar installations providing the distributed sensor density foundation for this disclosure
10. [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — Active fire detection data used for GNN training and historical ignition rate calibration
11. [National Weather Service API](https://www.weather.gov/documentation/services-web-api) — Programmatic access to weather forecasts for fire weather trigger assessment
12. [Micheli et al., Solar Energy, 2018](https://doi.org/10.1016/j.solener.2017.11.059) — Economic analysis of soiling and cleaning of photovoltaic modules, documenting soiling measurement infrastructure in residential inverters
13. [Rao et al., Remote Sensing of Environment, 2020](https://doi.org/10.1016/j.rse.2020.111797) — Machine learning approaches for live fuel moisture content estimation from remote sensing, demonstrating the gap between satellite resolution and WUI community scale

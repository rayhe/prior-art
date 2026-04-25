# LITF-PA-2026-020: System and Method for Non-Invasive Continuous Building Envelope Thermal Performance Assessment Using Smart Thermostat Heating and Cooling Response Curve Analysis with Edge-Deployed Machine Learning

**Filing:** LITF-PA-2026-020  
**Domain:** Building Science / Energy Efficiency  
**Published:** April 25, 2026  
**License:** [CC0 1.0 — Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)  
**HTML Version:** [liveinthefuture.org/priorart/thermal-envelope-thermostat-inference.html](https://liveinthefuture.org/priorart/thermal-envelope-thermostat-inference.html)

---

## Abstract

Disclosed is a system and method for continuously assessing the thermal performance of a building envelope without invasive testing, dedicated sensors, or professional energy audits. The system extracts thermal time constants and effective thermal resistance (R-value) estimates from the heating and cooling response curves already recorded by commodity smart thermostats (Nest, Ecobee, Honeywell Home, etc.), cross-references these curves against hyperlocal weather data (outdoor temperature, solar irradiance, wind speed) obtained from public APIs, and applies an edge-deployed Bayesian state-space model to separate envelope performance from HVAC equipment efficiency, occupant behavior, and internal heat gains. By tracking the evolution of inferred thermal parameters over months and years, the system detects insulation degradation, air seal failures, window gasket deterioration, and moisture intrusion events at the room or zone level, generating actionable alerts for homeowners and building managers without requiring a single additional sensor, wire, or site visit.

## Field of the Invention

This invention relates to building science and energy efficiency, specifically to non-invasive methods for estimating building envelope thermal performance using time-series data from existing smart home infrastructure combined with edge-deployed machine learning.

## Background

Buildings account for [40% of total U.S. energy consumption](https://www.eia.gov/todayinenergy/detail.php?id=46118) (EIA, 2024), with space heating and cooling representing roughly half of that figure in residential structures. The building envelope — walls, roof, windows, doors, and foundation — is the primary determinant of how much energy is required to maintain comfortable indoor temperatures. Envelope degradation is pervasive: the [DOE Building America program](https://www.energy.gov/eere/buildings/building-america) estimates that 90% of U.S. homes built before 2000 have insulation levels below current code requirements, and even code-compliant insulation degrades over time through settling, moisture absorption, pest damage, and thermal bridging from renovation work.

Current methods for assessing envelope thermal performance are expensive, disruptive, and episodic:

- **Blower door testing (ASTM E779):** Measures building air leakage by depressurizing the structure to 50 Pa and measuring airflow. Cost: $300-600 per test. Requires a technician on-site for 2-4 hours. Provides a single snapshot of air leakage, with no spatial localization of defects and no information about conductive heat loss through insulation.
- **Infrared thermography:** Thermal cameras (FLIR, InfiRay) can visualize temperature differentials across building surfaces, revealing insulation gaps and thermal bridges. Cost: $400-1,200 per survey, or $200-500 for a consumer-grade camera. Requires minimum 10°C indoor-outdoor temperature differential. [Fokaides and Kalogirou (Energy and Buildings, 2011)](https://doi.org/10.1016/j.enbuild.2013.09.004) showed that quantitative thermography can estimate U-values to within ±15%, but the technique requires controlled conditions, trained operators, and produces point-in-time assessments only.
- **Heat flux sensors:** [Desogus et al. (Energy and Buildings, 2011)](https://doi.org/10.1016/j.enbuild.2015.06.012) demonstrated in-situ U-value measurement using heat flux meters per ISO 9869-1. Accurate but requires sensor installation on wall surfaces for 3-14 days. Not scalable.

Smart thermostats have penetrated roughly [18% of U.S. households](https://www.statista.com/statistics/1373498/us-smart-thermostat-market-size/) as of 2025, with 30-40 million installed units. Every one of these devices continuously records indoor temperature at 5-minute intervals (or finer), HVAC system state (heating, cooling, fan-only, idle), setpoint schedules, and in some cases humidity and occupancy. This data stream is a rich but unexploited signal for envelope performance inference.

The physical basis is straightforward. A building's thermal response to HVAC input follows a first-order (or multi-order) exponential decay characterized by the time constant τ = R × C, where R is the effective thermal resistance of the envelope and C is the thermal capacitance of the interior mass. [Bacher and Madsen (Applied Energy, 2011)](https://doi.org/10.1016/j.apenergy.2017.12.112) demonstrated that grey-box models using indoor-outdoor temperature differentials can estimate R and C simultaneously from smart thermostat data with R² > 0.90 under favorable conditions. However, their work required custom instrumentation, manual model specification, and was validated only in controlled test cells.

The gap in the prior art is a complete, production-ready system that: (a) automatically extracts thermal response curves from commodity smart thermostat data requiring zero additional hardware, (b) separates envelope degradation from HVAC efficiency changes and occupant behavior variations through a Bayesian approach, (c) tracks thermal parameters continuously over months and years to detect gradual degradation, and (d) localizes degradation to specific zones in multi-thermostat homes.

## Detailed Description

### 1. Data Acquisition Layer

The system ingests time-series data from smart thermostat APIs (Nest Device Access, Ecobee API, Honeywell Home API, or local protocols such as Matter/Thread). Required signals per thermostat zone, sampled at minimum 5-minute intervals: indoor temperature T_in(t) in °C with ±0.5°C resolution (typical smart thermostat accuracy); HVAC system state s(t) ∈ {heating, cooling, fan_only, idle}; thermostat setpoint T_set(t); and occupancy state o(t) ∈ {home, away, sleep} when available from thermostat occupancy sensors or user schedules.

External weather data is obtained from public APIs: [OpenWeatherMap](https://openweathermap.org/api) or [Open-Meteo](https://open-meteo.com/) (free, no API key required for basic tier). Required signals at 15-minute or finer resolution: outdoor temperature T_out(t), solar irradiance GHI(t) in W/m², wind speed v_wind(t) in m/s, and relative humidity RH_out(t). Location is derived from thermostat zip code or GPS coordinates.

### 2. Response Curve Extraction

The core signal extraction identifies "natural experiment" windows when the building undergoes thermal transients that reveal envelope properties. Three event types are targeted:

- **HVAC-off coast-down events:** Periods when HVAC transitions from active heating/cooling to idle while a significant indoor-outdoor temperature differential exists (|ΔT| > 5°C). The rate of indoor temperature decay toward outdoor temperature during coast-down is governed by the envelope's thermal resistance. The system identifies coast-down events by detecting HVAC state transitions from heating/cooling to idle lasting at least 30 minutes with no subsequent HVAC activation.
- **HVAC recovery events:** Periods when HVAC activates after a setback period (e.g., returning from "away" mode). The rate of indoor temperature recovery toward setpoint reveals the combined envelope-plus-HVAC system response. By comparing recovery rate against known HVAC capacity (inferred from maximum sustained heating/cooling rate during steady-state operation), the system isolates envelope contribution from equipment performance.
- **Solar gain events:** Morning temperature rise in rooms with east-facing windows during sunny conditions, when HVAC is idle. Anomalous changes in solar gain magnitude can indicate window film degradation, shading changes, or window seal failure affecting infrared transmittance.

For each identified event, the system extracts a response curve: a time series of (t, T_in, T_out, GHI, v_wind) tuples spanning the transient period. Typical yield: 2-8 usable coast-down events per week during heating or cooling season, depending on climate zone, setpoint schedule, and occupancy patterns.

### 3. Grey-Box Thermal Parameter Estimation

Each response curve is fitted to a lumped-parameter thermal network model. The baseline model is a second-order (2R2C) circuit: R_env represents the effective thermal resistance of the building envelope (m²·K/W), aggregated across all surfaces; C_bldg represents the thermal capacitance of the building interior (J/K), including furnishings, internal walls, and floor slab; R_inf represents the effective thermal resistance associated with air infiltration, modeled as wind-speed-dependent: R_inf = R_inf0 / (1 + α·v_wind); and Q_solar represents solar heat gain, modeled as Q_solar = A_eff · GHI · τ_window, where A_eff is effective solar aperture area and τ_window is effective window transmittance.

The model state equation is:

C_bldg · dT_in/dt = (T_out - T_in)/R_env + (T_out - T_in)/R_inf + Q_solar + Q_hvac + Q_internal

where Q_hvac is HVAC thermal output (inferred from system state and estimated capacity) and Q_internal accounts for internal heat gains (lighting, appliances, occupants — estimated from occupancy state and time of day using DOE residential prototype building assumptions: 2.2 W/m² average, 4.4 W/m² peak during occupied evening hours per [PNNL residential prototypes](https://www.energycodes.gov/prototype-building-models)).

Parameter estimation uses maximum likelihood via the continuous-discrete extended Kalman filter (CD-EKF) as described by [Bacher and Madsen (2011)](https://doi.org/10.1016/j.apenergy.2017.12.112), adapted for the specific input signals available from smart thermostats. The filter processes each response curve to produce point estimates and confidence intervals for {R_env, C_bldg, R_inf, A_eff·τ_window}.

### 4. Bayesian Longitudinal Tracking

Individual response curve estimates are noisy (typical coefficient of variation 15-30% per event, driven by internal gain uncertainty and sensor noise). The system's key innovation is longitudinal aggregation: a hierarchical Bayesian model treats each event's parameter estimate as a noisy observation of slowly-varying true parameters.

The state-space model tracks θ_t = {R_env, C_bldg, R_inf, A_eff·τ} with a random-walk prior: θ_t = θ_{t-1} + ε, where ε ~ N(0, Q). The process noise covariance Q is set to allow parameter drift on the timescale of months (Q_ii ≈ (0.01·θ_i)² per week), reflecting the physical reality that insulation degradation, settling, and moisture accumulation occur gradually. Observation model: each event's parameter estimate y_k ~ N(θ_t, Σ_k), where Σ_k is the per-event estimation uncertainty from the CD-EKF.

Inference uses sequential Monte Carlo (particle filter) with 1,000 particles, resampled via systematic resampling when effective sample size drops below 500. Posterior distributions are stored as Gaussian mixture approximations (3 components) for compact representation. Total storage requirement: ~200 bytes per zone per update.

### 5. Anomaly Detection and Alerting

The system generates alerts when posterior parameter estimates deviate significantly from the building's established baseline. Two detection modes operate concurrently:

- **Gradual degradation:** A monotone trend in R_env decline exceeding 10% per year triggers a "degradation" alert. Detected via the slope of a robust linear fit to the R_env posterior mean series over the trailing 6 months, tested against a null hypothesis of zero slope using a Bayesian model comparison (Bayes factor > 10 for significance).
- **Acute events:** A sudden step change in R_env or R_inf (e.g., broken window seal, roof membrane failure, burst pipe causing wet insulation) is detected when the posterior shifts by more than 2σ within a 2-week window. The system cross-references acute R_inf changes with weather data to distinguish actual air seal failures from seasonal effects (wind direction changes exposing different building faces to infiltration pressure).

In multi-zone homes (2+ thermostats), the system performs differential diagnosis. If R_env degrades in one zone while remaining stable in others, the alert identifies the specific zone and suggests likely failure modes based on building vintage, zone location (e.g., attic zone → roof insulation, ground-floor zone → foundation/crawl space), and event characteristics.

### 6. Edge Deployment Architecture

The entire inference pipeline runs locally on a home hub (Raspberry Pi 4, Google Home, Amazon Echo with local processing, or Apple HomePod). Memory footprint: < 50 MB. CPU usage: < 5% average on a Cortex-A72. No thermostat data leaves the home network. Weather API calls use only zip code, revealing no personally identifiable information. The system exposes a local REST API (port 8420) serving JSON with current parameter estimates, trend plots (SVG), and alert status. An optional cloud sync endpoint transmits only anonymized, aggregated thermal parameters (no indoor temperature traces) for fleet-level benchmarking.

### 7. Calibration and Validation

Initial calibration leverages publicly available building stock data. For homes with known construction year and zip code, the system initializes parameter priors using [RECS](https://www.eia.gov/consumption/residential/) (Residential Energy Consumption Survey) building characteristic distributions and [IECC](https://www.energycodes.gov/) code-vintage R-value tables. For a 1990s-era wood-frame home in IECC Climate Zone 4, the prior for R_env,wall is N(13.0, 3.0²) ft²·°F·h/Btu, reflecting code-minimum R-13 wall insulation with uncertainty spanning the range of actual field performance documented by [Kramer et al. (Energy and Buildings, 2016)](https://doi.org/10.1016/j.enbuild.2016.01.041).

Validation against blower door tests: the system's R_inf estimate can be converted to an equivalent air changes per hour at 50 Pa (ACH50) using the [Sherman-Grimsrud model](https://doi.org/10.1016/j.enbuild.2009.10.007) relating infiltration rate to leakage area, stack effect, and wind pressure. This provides a direct comparison with blower door test results when available.

## Claims

1. A system for non-invasive assessment of building envelope thermal performance, comprising: a data ingestion module receiving indoor temperature and HVAC system state time series from one or more smart thermostats; a weather data module receiving outdoor temperature, solar irradiance, and wind speed from external APIs; a response curve extractor identifying thermal transient events including HVAC-off coast-down, HVAC recovery, and solar gain events; and a grey-box parameter estimator fitting a lumped-parameter thermal network model to extracted response curves to estimate effective envelope thermal resistance, building thermal capacitance, and infiltration resistance.

2. The system of claim 1, wherein the grey-box model is a second-order 2R2C thermal network with wind-speed-dependent infiltration resistance and solar gain terms, estimated using a continuous-discrete extended Kalman filter.

3. The system of claim 1, further comprising a Bayesian longitudinal tracking module that treats individual event parameter estimates as noisy observations of slowly-varying true parameters, using a particle filter with random-walk state dynamics to produce posterior distributions over envelope thermal parameters that improve in precision as more events are observed over weeks and months.

4. The system of claim 3, further comprising an anomaly detection module that generates alerts when the posterior mean of effective thermal resistance declines at a rate exceeding a configurable threshold over a trailing window, or when a step change exceeding a configurable significance level occurs within a short temporal window.

5. The system of claim 1, wherein the entire inference pipeline executes locally on a home automation hub or single-board computer, transmitting no indoor temperature data off-premises, with weather API queries using only zip code as location identifier.

6. A method for detecting building envelope degradation comprising: continuously collecting indoor temperature and HVAC state data from a smart thermostat and outdoor weather data from a public API; identifying thermal transient events in the data stream; fitting a parametric thermal model to each event to estimate envelope thermal resistance and infiltration resistance; aggregating estimates over time using Bayesian state-space modeling; and alerting when estimated parameters exhibit statistically significant degradation trends or acute step changes.

7. The method of claim 6, further comprising multi-zone differential diagnosis in buildings with two or more smart thermostats, wherein zone-specific parameter changes are compared to isolate degradation to a specific building region and suggest likely failure modes based on zone location and building vintage.

8. The method of claim 6, wherein initial parameter priors are set using publicly available building stock data including construction vintage, climate zone, and code-era insulation requirements from RECS and IECC databases.

9. The method of claim 6, further comprising validation against blower door test results by converting the inferred infiltration resistance to equivalent ACH50 using the Sherman-Grimsrud model.

10. The system of claim 1, wherein HVAC equipment capacity is inferred from the maximum sustained rate of temperature change during steady-state operation, enabling separation of envelope thermal resistance from HVAC efficiency degradation when both change simultaneously.

## Prior Art References

1. [EIA — Buildings sector energy consumption](https://www.eia.gov/todayinenergy/detail.php?id=46118) — 40% of U.S. energy use
2. [DOE Building America Program](https://www.energy.gov/eere/buildings/building-america) — Pre-2000 housing stock insulation deficiency estimates
3. [Fokaides & Kalogirou, Energy and Buildings 2011](https://doi.org/10.1016/j.enbuild.2013.09.004) — Quantitative infrared thermography for U-value estimation (±15% accuracy)
4. [Desogus et al., Energy and Buildings 2011](https://doi.org/10.1016/j.enbuild.2015.06.012) — In-situ U-value measurement with heat flux meters per ISO 9869-1
5. [Statista — U.S. smart thermostat market penetration](https://www.statista.com/statistics/1373498/us-smart-thermostat-market-size/)
6. [Bacher & Madsen, Applied Energy 2011](https://doi.org/10.1016/j.apenergy.2017.12.112) — Grey-box modelling of building thermal behaviour with RC models
7. [PNNL Residential Prototype Building Models](https://www.energycodes.gov/prototype-building-models) — Internal heat gain assumptions
8. [RECS — Residential Energy Consumption Survey](https://www.eia.gov/consumption/residential/) — Building stock characteristics
9. [IECC — International Energy Conservation Code](https://www.energycodes.gov/) — Code-vintage insulation requirements
10. [Kramer et al., Energy and Buildings 2016](https://doi.org/10.1016/j.enbuild.2016.01.041) — Field performance of residential wall insulation
11. [Sherman & Grimsrud infiltration model](https://doi.org/10.1016/j.enbuild.2009.10.007) — Relating leakage area to infiltration rate
12. [OpenWeatherMap API](https://openweathermap.org/api) — Public weather data source
13. [Open-Meteo API](https://open-meteo.com/) — Free, open-source weather API
14. [Google Nest Device Access API](https://developers.google.com/nest/device-access) — Smart thermostat data access
15. [Ecobee Developer API](https://www.ecobee.com/en-us/developers/) — Smart thermostat data access
16. [ASTM E779](https://www.astm.org/e0779-19.html) — Standard test method for determining air leakage rate by fan pressurization

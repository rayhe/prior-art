# System and Method for Continuous Crop Water Stress Detection and Precision Irrigation Scheduling via Contactless Monitoring of Plant Xylem Cavitation Acoustic Emissions Using Ultrasonic MEMS Microphone Arrays and Edge-Deployed Neural Networks

**LITF-PA-2026-013 · AgTech / Precision Irrigation**
**Published:** 2026-04-15
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for continuously detecting crop water stress at the individual plant level by monitoring ultrasonic acoustic emissions (AE) produced by xylem cavitation events. When a plant experiences water deficit, the tension in its xylem conduits exceeds a species-specific threshold, causing dissolved gas to nucleate and form embolisms that produce characteristic ultrasonic clicks in the 20–250 kHz frequency range. The system deploys weatherproof sensor nodes, each containing an array of two to four wideband MEMS ultrasonic microphones (bandwidth 10–80 kHz, e.g., Knowles SPU0410LR5H-QB) positioned 5–15 cm from the plant stem without physical contact. An edge microcontroller (ESP32-S3 or nRF5340) samples the acoustic signal at 192 kHz, extracts time-domain pulse features (amplitude, duration, rise time, inter-event interval) and frequency-domain features (spectral centroid, peak frequency, bandwidth), and feeds them to a lightweight 1D temporal convolutional network (TCN) that classifies water stress into four levels: no stress, mild (pre-visible), moderate, and severe. The system correlates AE event rates with environmental context (air temperature, vapor pressure deficit, solar radiation, soil water potential from optional tensiometers) to separate drought-induced cavitation from temperature-driven diurnal patterns. Aggregated per-plant stress scores drive a zone-level irrigation scheduling engine that triggers watering only when and where plants actually need it, targeting 20–40% water savings compared to timer-based or soil-moisture-only irrigation. All inference runs on the sensor node itself at under 10 mW average power, enabling solar-powered deployment across orchards, vineyards, and row crops without wired infrastructure.

## Field of the Invention

This invention relates to precision agriculture and irrigation management, specifically to the use of contactless ultrasonic acoustic emission monitoring of plant xylem cavitation events combined with on-device machine learning inference for real-time, per-plant water stress detection and automated irrigation scheduling.

## Background

Agriculture consumes approximately 42% of all freshwater withdrawals in the United States ([USGS, 2020](https://www.usgs.gov/mission-areas/water-resources/science/irrigation-water-use)), totaling 118 billion gallons per day. Globally, irrigation accounts for 70% of freshwater withdrawals ([FAO AQUASTAT](https://www.fao.org/aquastat/en/)). With groundwater depletion accelerating in major agricultural basins — the [NASA GRACE satellite mission](https://grace.jpl.nasa.gov/) has documented alarming declines in the Central Valley, Ogallala, and Indo-Gangetic aquifers — precision irrigation is no longer optional. It is an existential requirement for sustained food production.

The fundamental problem is that most irrigation systems water on schedules or in response to soil conditions, rather than in response to the plant's actual physiological need. A plant experiencing water stress shows measurable physiological changes (stomatal closure, reduced transpiration, loss of turgor, xylem embolism) hours to days before visible wilting. By the time a farmer sees leaf curl, significant yield loss has already occurred. Conversely, many irrigation systems overwater because they lack the resolution to know which zones or individual plants are adequately hydrated.

Current methods for detecting plant water stress include:

- **Soil moisture sensors (TDR, capacitance, tensiometers):** Measure water availability in the root zone, not in the plant itself. Soil moisture is a necessary but insufficient proxy: two adjacent plants in the same soil can have different stress levels due to root architecture, canopy exposure, or disease. Sensor placement is critical and unrepresentative. A single sensor covers 10–50 cm radius. Typical cost: $100–$500/sensor plus datalogger.
- **Infrared thermometry and thermal imaging:** Stressed plants close stomata, reducing transpiration and raising canopy temperature by 1–5°C. Infrared thermometers (Apogee SI-111, $200) measure single-point canopy temperature. Thermal drones (DJI Mavic 3T, $4,500+) provide field-scale maps but require flight planning, clear-sky conditions, and post-processing. Neither provides continuous, real-time data.
- **Dendrometers (stem diameter gauges):** Continuous sensors (Ecomatik, Plantsens) that measure micrometer-scale stem shrinkage/swelling as a proxy for water status. Accurate but expensive ($300–$800/sensor), require physical attachment to the stem (potential wound entry), and are impractical for row crops or plants with thin stems.
- **Pressure chamber (Scholander bomb):** The gold standard for measuring xylem water potential. A leaf is excised, sealed in a pressure chamber, and pressurized until xylem sap appears at the cut surface. The balancing pressure equals the (negative) xylem water potential. Accurate to ±0.05 MPa but destructive, labor-intensive (2–5 minutes per sample), and provides a snapshot rather than continuous monitoring.
- **Hyperspectral remote sensing:** Satellite or drone-mounted hyperspectral cameras detect water stress from spectral reflectance changes (chlorophyll absorption, water absorption bands). Spatial resolution from satellites (Sentinel-2: 10m) is too coarse for individual plants. Drones improve resolution but have the same temporal limitations as thermal imaging.

The scientific basis for acoustic emission monitoring of water stress has been established for decades. [Tyree and Sperry (1989)](https://doi.org/10.1146/annurev.pp.40.060189.000335) demonstrated that xylem cavitation produces detectable acoustic emissions, with event rates correlating to the percentage loss of hydraulic conductivity. More recently, [Khait et al. (2023)](https://doi.org/10.1016/j.cell.2023.03.009) published in *Cell* that stressed tomato and tobacco plants emit airborne ultrasonic sounds at frequencies of 20–100 kHz, detectable by microphones at distances up to several meters, with drought-stressed plants producing 25–35 clicks per hour compared to <1 per hour for well-watered plants. [De Roo et al. (2016)](https://doi.org/10.1111/nph.14285) showed that acoustic emission monitoring can detect the onset of xylem embolism formation in real-time.

The gap in the art is a complete, field-deployable system that: (a) uses contactless ultrasonic microphones rather than contact transducers (no physical attachment to the plant), (b) runs real-time cavitation event detection and stress classification entirely on an edge device, (c) separates drought-induced cavitation from environmental confounders (temperature, wind, insects), and (d) closes the loop to an irrigation controller for automated, per-zone scheduling based on plant physiological signals.

## Detailed Description

### 1. Physical Principle: Xylem Cavitation as a Water Stress Biomarker

Plants transport water from roots to leaves through xylem conduits (tracheids in conifers, vessels in angiosperms) under negative pressure (tension). This tension is generated by transpiration at the leaf surface. Under well-watered conditions, xylem tension in a grapevine typically ranges from -0.2 to -0.8 MPa. As soil dries and atmospheric demand increases, xylem tension rises (becomes more negative). When tension exceeds a species-specific threshold — approximately -1.0 to -1.5 MPa for *Vitis vinifera* (grapevine) — dissolved gas nucleates at pit membrane pores or pre-existing nanobubbles, forming an embolism that blocks water transport through the affected conduit. This is cavitation.

Each cavitation event releases elastic energy stored in the stretched water column and the conduit walls. This energy propagates as an acoustic wave through the plant tissue and into the surrounding air. The acoustic signature has several measurable characteristics: a sharp onset (rise time 5–50 μs), peak frequencies typically in the 20–250 kHz range (with most energy between 40–150 kHz for woody species), pulse durations of 50–500 μs, and amplitudes that decrease with distance following an inverse-square law modified by the plant's tissue attenuation. Critically, the *rate* of cavitation events is a direct, real-time indicator of ongoing hydraulic failure, making it the most physiologically direct measure of water stress available without destructive sampling.

### 2. Sensor Node Hardware

Each sensor node is a self-contained, weatherproof unit (IP65-rated enclosure, approximately 80 × 60 × 40 mm) designed for permanent outdoor deployment:

- **Ultrasonic microphone array:** Two to four Knowles SPU0410LR5H-QB analog MEMS microphones (sensitivity -42 dB re 1V/Pa, frequency response 100 Hz – 80 kHz, noise floor 33 dBA). The microphones are mounted in a semi-circular arc on a bracket that positions them 5–15 cm from the target stem, oriented inward. The multi-microphone configuration enables beamforming for noise rejection and rough source localization along the stem axis.
- **Analog front end:** Low-noise instrumentation amplifier (gain 40 dB) followed by a 4th-order Butterworth bandpass filter (15 kHz – 80 kHz cutoff) to reject audible-range environmental noise (wind, machinery, animals) while passing the cavitation emission band. The filter's -3 dB point at 15 kHz provides >60 dB rejection of frequencies below 5 kHz.
- **ADC and microcontroller:** ESP32-S3 (dual-core 240 MHz, 512 KB SRAM, integrated 12-bit ADC at 2 MSPS, WiFi/BLE) or Nordic nRF5340 (dual-core ARM Cortex-M33 at 128 MHz, lower power). The ADC samples at 192 kHz per channel (above Nyquist for the 80 kHz microphone bandwidth), capturing the full cavitation pulse waveform.
- **Environmental sensors:** BME280 temperature/humidity/pressure sensor (±1°C, ±3% RH); Si1145 UV/ambient light sensor for solar radiation estimation. Optional: soil tensiometer interface (analog 4-20 mA input) for co-located soil water potential measurement.
- **Power:** 3.7V 3000 mAh LiPo battery with 2W solar panel (80 × 60 mm monocrystalline). Average power consumption: 8 mW in standby (listening for events), 45 mW during active inference, <10 mW average. Battery lasts >30 days without sun; with solar, indefinite operation.
- **Communication:** LoRa (SX1262, 915 MHz ISM) for long-range, low-power data transmission to a field gateway. Per-plant stress scores, event counts, and environmental data are transmitted every 15 minutes in a 50-byte payload. LoRa range: 2–10 km line-of-sight, adequate for most agricultural settings.

Total BOM cost at volume (5,000+ units): microphones ($1.50 × 4 = $6), MCU ($3), analog front end ($1.50), environmental sensors ($1.50), LoRa module ($3), solar panel ($4), battery ($3), PCB + enclosure ($5), mounting bracket ($2). Total: approximately $29. Target retail including margin: $49–$79 per node.

### 3. Signal Processing Pipeline

The signal processing runs entirely on the edge microcontroller in three stages:

**Stage 1: Event Detection.** A continuous threshold detector monitors the filtered ultrasonic signal envelope. A cavitation event is detected when the instantaneous amplitude exceeds a background-adaptive threshold (set at 6σ above the rolling 10-second noise floor). The threshold adapts to ambient conditions — wind-induced noise raises the floor, quiet nights lower it. When a threshold crossing is detected, the system captures a 2 ms waveform window (384 samples at 192 kHz) centered on the event peak.

**Stage 2: Feature Extraction.** From each captured event waveform, the system extracts 12 features:
- **Time-domain (5):** peak amplitude, rise time (10–90% of peak), pulse duration (-6 dB envelope width), energy (integral of squared amplitude), inter-event interval (time since previous event).
- **Frequency-domain (5):** peak frequency, spectral centroid, bandwidth (interquartile range), spectral rolloff (95th percentile frequency), spectral flatness (Wiener entropy).
- **Array-derived (2):** inter-microphone time delay of arrival (TDOA, microsecond resolution, providing source bearing), coherence between microphone pairs (high coherence indicates localized source, low coherence indicates diffuse noise).

Feature extraction requires approximately 0.3 ms per event on ESP32-S3, supporting real-time processing at event rates up to 3,000 events/second (far exceeding physiological rates).

**Stage 3: Stress Classification.** A lightweight 1D temporal convolutional network (TCN) processes a sliding window of the most recent 60 minutes of event features (typically 0–2,000 events depending on stress level). The TCN architecture uses 4 dilated causal convolutional blocks (dilation factors 1, 2, 4, 8), each with 32 filters of kernel size 3, followed by global average pooling and a 4-class softmax output:
- **Level 0 (No stress):** 0–2 events/hour, consistent with thermal equilibrium noise.
- **Level 1 (Mild stress):** 3–15 events/hour, onset of cavitation at most vulnerable conduits. No visible symptoms. Irrigation within 24 hours prevents yield loss.
- **Level 2 (Moderate stress):** 16–60 events/hour, significant embolism formation. Stomatal closure measurable. Irrigation within 6 hours recommended.
- **Level 3 (Severe stress):** >60 events/hour, rapid hydraulic failure. Visible wilting imminent. Immediate irrigation required.

The TCN also receives the environmental context vector (temperature, humidity, VPD, solar radiation, hour of day) as a conditioning input, concatenated after the convolutional blocks. This is critical because cavitation rates follow a diurnal pattern even in well-watered plants: peak transpiration demand around solar noon generates transient tension peaks that cause minor cavitation in the most vulnerable conduits, recovering overnight as root pressure refills embolized conduits. The environmental context allows the model to distinguish this normal diurnal pattern from progressive drought-induced cavitation.

Model size: 45 KB (INT8 quantized). Inference time: 12 ms on ESP32-S3. The TCN runs once per minute, consuming negligible energy.

### 4. Irrigation Control Integration

The field gateway (a LoRa-to-WiFi/cellular bridge, e.g., RAK7268 or Dragino LPS8N) aggregates stress scores from all sensor nodes and computes zone-level irrigation decisions:

- Each irrigation zone contains N sensor nodes (e.g., 10 nodes monitoring 10 representative vines in a vineyard block of 500 vines).
- The zone stress score is the 75th percentile of individual node stress levels (using the 75th percentile rather than the mean ensures that the most stressed plants in the zone trigger irrigation, not just the average).
- When the zone stress score reaches Level 1, the system queues an irrigation event for the next irrigation window (typically nighttime to minimize evaporation).
- When the zone stress score reaches Level 2, irrigation is triggered immediately regardless of time.
- Irrigation duration is proportional to the accumulated stress-time integral (a zone at Level 2 for 4 hours receives more water than one that just crossed the Level 2 threshold).

The gateway communicates with commercial irrigation controllers via standard interfaces: Modbus RTU for commercial agricultural systems (Hunter ICC2, Rain Bird ESP-LXME), WiFi API for consumer/prosumer controllers (Rachio, Hydrawise), or dry-contact relay closure for legacy solenoid valve control. The closed-loop system continuously monitors cavitation rates during and after irrigation, confirming that the applied water reduced stress levels (cavitation rate should drop within 30–90 minutes of irrigation reaching the root zone).

### 5. Species Calibration and Transfer Learning

Cavitation characteristics vary substantially between plant species. Grapevine emits lower-frequency, higher-amplitude pulses than almond. Tomato cavitates at lower tensions than olive. The system uses a species-specific calibration approach:

A base model is pre-trained on laboratory data from 15 common crop species, using controlled dehydration experiments where plants are instrumented with both the acoustic sensor array and reference psychrometer measurements of xylem water potential. This provides ground-truth stress labels correlated with acoustic signatures.

For field deployment on a species not in the training set, the system runs a 7-day self-supervised calibration: it observes the diurnal pattern of acoustic emissions under the site's normal irrigation regime, using the daily cycle of emission onset (correlated with peak VPD) and cessation (correlated with nighttime recovery) to establish baseline event rates. Fine-tuning adjusts only the final classification layer's thresholds, preserving the learned event detection and feature extraction from the base model.

### 6. Noise Rejection

Outdoor agricultural environments are acoustically noisy. The system employs four noise rejection mechanisms:

1. **Bandpass filtering** (hardware): The 15–80 kHz passband rejects most environmental noise (wind: <5 kHz, machinery: <10 kHz, birds: <12 kHz). Insects (particularly cicadas and katydids) can produce ultrasonic sound but with continuous, narrowband signatures easily distinguishable from the impulsive, broadband cavitation events.
2. **Array coherence gating** (firmware): A true cavitation event from the nearby stem produces correlated arrivals across all microphones with consistent TDOA. Diffuse noise (wind, rain) produces low inter-microphone coherence (<0.3). Events with coherence below 0.6 are rejected.
3. **Waveform template matching** (firmware): Cavitation pulses have a characteristic asymmetric shape (fast rise, exponential decay). A matched filter correlates each detected event against a species-specific template, rejecting events with correlation <0.5.
4. **Contextual filtering** (model): The TCN learns to suppress rain events (which produce high event rates but with distinctive spectral signatures and high correlation with the on-board humidity sensor's rain detection).

### 7. Figures Description

- **Figure 1:** System architecture showing sensor node mounted near a grapevine stem, LoRa transmission to field gateway, gateway processing zone-level decisions, and command output to irrigation controller.
- **Figure 2:** Xylem cross-section diagram showing vessel elements under normal tension and during cavitation, with acoustic emission wavefront radiating from the embolism site.
- **Figure 3:** Example 24-hour acoustic emission timeline from a grapevine under progressive drought, showing the transition from Level 0 (night) through Level 1 (midday) to Level 2 (afternoon), with corresponding VPD overlay.
- **Figure 4:** Cavitation event waveform (2 ms window) showing characteristic fast-rise, exponential-decay pulse shape, with annotated features: peak amplitude, rise time, pulse duration, and spectral content inset.
- **Figure 5:** TCN architecture diagram showing 1D dilated causal convolutional blocks, environmental context concatenation, and 4-class stress output.
- **Figure 6:** Zone-level irrigation control flow diagram showing sensor node aggregation, 75th-percentile stress scoring, irrigation triggering rules, and closed-loop confirmation via post-irrigation cavitation rate monitoring.

## Claims

1. A system for detecting water stress in a living plant, comprising: one or more ultrasonic microphones positioned at a distance from a plant stem without physical contact with the plant; an analog signal conditioning circuit that bandpass-filters the microphone output to pass frequencies in the range characteristic of xylem cavitation acoustic emissions; an edge computing device that detects individual cavitation events from the filtered signal and classifies the plant's water stress level based on event rate, event spectral characteristics, and environmental context using a trained neural network; wherein the system operates without any sensor physically attached to the plant tissue.

2. The system of claim 1, wherein the ultrasonic microphones are MEMS microphones with a frequency response extending to at least 60 kHz, positioned 5–20 cm from the plant stem, and the analog conditioning circuit includes a bandpass filter with a lower cutoff frequency of at least 10 kHz to reject audible-range environmental noise.

3. The system of claim 1, wherein the neural network is a temporal convolutional network that ingests a time series of extracted event features over a sliding window of at least 30 minutes and classifies water stress into at least three severity levels, wherein the event features include at least: peak amplitude, rise time, pulse duration, peak frequency, spectral centroid, and inter-event interval.

4. The system of claim 1, wherein the neural network further receives environmental context inputs comprising at least air temperature and vapor pressure deficit, enabling the model to distinguish drought-induced cavitation from diurnal transpiration-driven cavitation events that occur in well-watered plants during peak evaporative demand periods.

5. The system of claim 1, further comprising an array of at least two ultrasonic microphones per node, wherein inter-microphone coherence and time-delay-of-arrival are used to reject diffuse acoustic noise and confirm that detected events originate from the monitored plant stem rather than from environmental sources.

6. The system of claim 1, further comprising a wireless communication link between the sensor node and an irrigation controller, wherein the irrigation controller triggers watering for the zone containing the sensor node when the classified stress level exceeds a configurable threshold, and wherein the system monitors cavitation rates after irrigation to confirm that the applied water reduced plant stress.

7. A method for precision irrigation scheduling comprising: positioning one or more contactless ultrasonic microphones near a plant stem; continuously monitoring the ultrasonic acoustic emission signal for impulsive events characteristic of xylem cavitation; extracting time-domain and frequency-domain features from each detected event; classifying the plant's water stress level using a machine learning model that considers both event characteristics and environmental context; and triggering irrigation for the zone containing the plant when the classified stress level exceeds a threshold, wherein irrigation duration is proportional to a stress-time integral that accumulates based on the severity and duration of detected stress.

8. The method of claim 7, wherein a zone stress score is computed from multiple sensor nodes within the same irrigation zone by taking the 75th percentile of individual node stress levels, ensuring that the most stressed plants in the zone determine the irrigation decision.

9. The method of claim 7, further comprising a species-specific calibration procedure wherein a base model pre-trained on laboratory cavitation data from multiple crop species is fine-tuned during an initial field observation period by using the diurnal pattern of acoustic emissions under the site's existing irrigation regime to establish baseline event rates, enabling deployment on crop species not represented in the original training dataset.

10. The system of claim 1, wherein the edge computing device operates at an average power consumption below 15 milliwatts, is powered by a solar panel and rechargeable battery, requires no wired power or data connections, and communicates wirelessly via a low-power wide-area network protocol, enabling deployment of one sensor node per individual plant across orchards and vineyards without infrastructure modification.

## Implementation Notes

The primary deployment target is high-value perennial crops where individual plant health directly affects yield and quality: wine grapes ($10,000–$50,000/acre revenue), almonds ($5,000–$8,000/acre), pistachios ($8,000–$12,000/acre), avocados, and citrus. At $49–$79 per sensor node and one node per 5–20 plants (depending on canopy uniformity), the per-acre sensor cost is $50–$300, which is recoverable within one season from water savings alone in drought-priced California water districts ($800–$1,500/acre-foot).

The system is a complement to, not a replacement for, soil moisture monitoring. Soil sensors answer "is water available in the root zone?" while xylem cavitation monitoring answers "is the plant actually stressed?" The combination provides the most complete picture: a plant may show stress despite adequate soil moisture (due to root damage, salt stress, or disease restricting uptake), or may be well-hydrated despite low soil moisture readings (deep root access to water table).

A key limitation is that the current ultrasonic MEMS microphones (80 kHz bandwidth) capture only the lower-frequency portion of the cavitation emission spectrum. Many woody species produce peak emission energy at 100–200 kHz, above the microphone's sensitivity range. The system detects these events through their lower-frequency spectral tail (still above the noise floor at 40–60 kHz), but with reduced signal-to-noise ratio compared to what a purpose-built 200+ kHz ultrasonic transducer would achieve. As MEMS microphone bandwidth improves, the system's detection sensitivity and species range will expand.

## Prior Art References

1. [Tyree, M.T. and Sperry, J.S. (1989). "Vulnerability of Xylem to Cavitation and Embolism."](https://doi.org/10.1146/annurev.pp.40.060189.000335) Annual Review of Plant Physiology and Plant Molecular Biology, 40, 19–38. Foundational work establishing acoustic emission detection of xylem cavitation.
2. [Khait, I. et al. (2023). "Sounds emitted by plants under stress are airborne and informative."](https://doi.org/10.1016/j.cell.2023.03.009) Cell, 186(7), 1328–1336. Key finding that stressed plants emit airborne ultrasonic sounds detectable by microphones at distance.
3. [De Roo, L. et al. (2016). "Acoustic Emission analysis of drought-induced cavitation in stems of two deciduous broad-leaved tree species."](https://doi.org/10.1111/nph.14285) New Phytologist, 213(2), 574–585. Real-time acoustic emission monitoring of xylem embolism formation in trees.
4. [USGS. "Irrigation Water Use."](https://www.usgs.gov/mission-areas/water-resources/science/irrigation-water-use) U.S. Geological Survey. Agricultural water consumption data (42% of U.S. freshwater withdrawals).
5. [FAO. "AQUASTAT Main Database."](https://www.fao.org/aquastat/en/) Food and Agriculture Organization. Global irrigation water withdrawal statistics (70% of freshwater).
6. [NASA. "GRACE-FO: Gravity Recovery and Climate Experiment Follow-On."](https://grace.jpl.nasa.gov/) Groundwater depletion monitoring from space.
7. [Scholander, P.F. et al. (1965). "Sap Pressure in Vascular Plants."](https://doi.org/10.1126/science.148.3668.339) Science, 148(3668), 339–346. The pressure chamber method for measuring xylem water potential.
8. [Jones, H.G. (2004). "Irrigation scheduling: advantages and pitfalls of plant-based methods."](https://doi.org/10.1093/jxb/erh213) Journal of Experimental Botany, 55(407), 2427–2436. Review of plant-based irrigation scheduling approaches.
9. [US20200132654A1](https://patents.google.com/patent/US20200132654A1/en). "System for monitoring soil conditions based on acoustic data." Monitors soil acoustic properties during planting; does not monitor plant tissue acoustic emissions.
10. [US7363113B2](https://patents.google.com/patent/US7363113B2/en). "Method and system for irrigation control." Uses soil moisture and weather data for irrigation scheduling; no plant physiological sensing.
11. [Knowles SPU0410LR5H-QB.](https://www.knowles.com/docs/default-source/default-document-library/spu0410lr5h-qb-revh.pdf) MEMS microphone datasheet. Frequency response 100 Hz – 80 kHz.
12. [Espressif ESP32-S3.](https://www.espressif.com/en/products/socs/esp32-s3) System-on-chip datasheet. Dual-core 240 MHz, 512 KB SRAM, WiFi/BLE, 12-bit ADC.
13. [Vergeynst, L.L. et al. (2015). "Cavitation: a blessing in disguise? New method to establish vulnerability curves and assess hydraulic capacitance of woody tissues."](https://doi.org/10.1093/treephys/tpv002) Tree Physiology, 35(4), 400–409. Acoustic emission methods for vulnerability curve construction.
14. [NASS. "Farm and Ranch Irrigation Survey (2018)."](https://www.nass.usda.gov/Surveys/Guide_to_NASS_Surveys/Farm_and_Ranch_Irrigation/) USDA National Agricultural Statistics Service. Irrigation practices and water use statistics.

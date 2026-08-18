# PA-2026-142: System and Method for Automated Residential Pest Species Identification and Infestation Density Estimation Using Smart Doorbell and Security Camera Near-Infrared Phototactic Capture Imagery with Edge-Deployed Entomological Classification

**Filing:** LITF-PA-2026-142  
**Domain:** Pest Management / Computer Vision / Edge AI  
**Published:** August 17, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for automated identification and density estimation of pest insect species at residential properties by repurposing the near-infrared (NIR) illumination and imaging hardware already present in consumer smart doorbells and outdoor security cameras. Nocturnal and crepuscular insects exhibit positive phototaxis toward NIR light sources in the 850-940 nm wavelength range commonly used by these devices, causing them to congregate within the camera's field of view in predictable patterns. The system captures timestamped image frames during NIR-illuminated nighttime operation, applies background subtraction and morphological filtering to isolate individual insect silhouettes, extracts discriminative features including body length, wing venation aspect ratio, antennae geometry, and flight trajectory kinematics, and classifies each detection to the family or genus level using a quantized convolutional neural network running on the camera's edge processor. Aggregated nightly species counts, phenological trend tracking, and spatial correlation across neighboring cameras enable early detection of emerging infestations (termite swarmers, carpenter ant alates, bark beetles, pantry moths, mosquito population surges) before structural damage occurs or disease vectors reach epidemiologically significant density thresholds. The system requires no additional hardware, no pesticide-based traps, and no manual specimen collection.

## Field of the Invention

This invention relates to integrated pest management and public health entomology, specifically to passive optical surveillance of pest insect populations using the near-infrared imaging capabilities of existing consumer security camera infrastructure combined with edge-deployed machine learning for species-level classification and population density estimation.

## Background

Structural pest damage to residential buildings in the United States costs homeowners an estimated $5 billion annually for termites alone (National Pest Management Association), with additional billions attributable to carpenter ants, powder post beetles, and wood-boring insects. Early detection is the single most important factor in limiting structural damage: a termite colony typically forages for 3-5 years before producing visible signs such as mud tubes or frass, by which point structural compromise is already substantial. The median cost of termite remediation after detection of visible damage is $3,000-$5,000, compared to $500-$800 when colonies are intercepted during swarming events before establishment.

Vector-borne disease surveillance faces parallel challenges. Mosquito-borne illness (West Nile virus, Eastern Equine Encephalitis, Zika) causes 700+ deaths annually in the US (CDC Vector-Borne Disease Division). Public health mosquito surveillance relies on CO2-baited light traps (CDC Miniature Light Trap, ~$200/unit, manual servicing every 24-48 hours) deployed at 30-50 sites per county. This sparse, episodic sampling misses hyperlocal breeding hotspots in individual yards.

Current residential pest detection methods have significant limitations:

- **Professional inspections:** Annual termite inspections ($75-$150/visit) are visual and rely on inspector experience. Detection rates for active subterranean termite infestations during visual inspection average 72% sensitivity (Rust & Su, Annual Review of Entomology, 2012). The 28% miss rate means nearly one in three active infestations goes undetected per inspection cycle.
- **Bait stations:** Sentricon and similar in-ground bait monitoring systems ($1,500-$3,000 installed, $300/year monitoring) detect termites only after foragers discover and feed on bait matrix, which can take 6-18 months. Stations sample at 10-15 points around a structure's perimeter, missing foraging corridors between stations.
- **Sticky traps and pheromone lures:** Effective for specific pests (pantry moths, cockroaches) in indoor settings. Require manual inspection, provide species-specific rather than broad-spectrum monitoring, and are aesthetically objectionable to many homeowners. No automated counting or species identification.
- **Acoustic emission monitoring:** Mankin et al. (Florida Entomologist, 2002) demonstrated acoustic detection of termite feeding activity in wood. Dedicated hardware costs $2,000-$5,000 per installation. Limited to species that produce detectable feeding sounds within monitored structural members.

Meanwhile, the installed base of NIR-equipped outdoor cameras is massive and growing. As of 2025, an estimated 85 million smart doorbells and outdoor security cameras are deployed at US residences (Statista, 2025). Ring alone reported 10+ million active devices in 2023. Every one of these cameras has a 850 nm or 940 nm NIR LED array and a CMOS sensor with the IR-cut filter mechanically removed during night mode. Every one of them records video of insects attracted to their light every single night. That data is currently discarded as nuisance motion events.

Insect phototaxis toward artificial light is among the most extensively documented phenomena in entomology. Owens et al. (Philosophical Transactions B, 2020) reviewed 229 studies confirming that the majority of nocturnal insect orders exhibit strong positive phototaxis, with peak attraction in the UV-A and near-infrared bands. Critically, the taxonomic composition of insects attracted to a given light source is a reliable, reproducible estimator of local species assemblage: van Grunsven et al. (Insect Conservation and Diversity, 2014) showed that automated light trap counts correlate with professional survey estimates at r = 0.82-0.91 for major pest families.

The gap in the prior art is a complete system that: (a) repurposes the NIR imaging hardware already deployed at tens of millions of homes for entomological surveillance without any additional sensors, traps, or consumables; (b) classifies pest insects to genus or family level using edge inference on the camera's existing processor; (c) tracks nightly population trends to detect swarming events, seasonal emergence shifts, and infestation onset; and (d) correlates detections across neighboring cameras to identify breeding hotspots, dispersal corridors, and neighborhood-scale infestation fronts.

## Detailed Description

### 1. Optical Capture Geometry and Insect Imaging Characteristics

Smart doorbells and security cameras mount at heights of 1.0-3.5 meters above grade, directing their field of view toward approach paths, driveways, and yards. The NIR LED illuminator array (typically 6-12 LEDs at 850 nm or 940 nm, total radiant flux 500-2,000 mW) creates an illumination cone with effective insect-attraction range of 3-15 meters depending on power, species, and ambient darkness. Insects exhibiting positive phototaxis approach the light source and fly or crawl within the camera's depth of field.

At the sensor, nocturnal insects produce characteristic signatures depending on distance, size, and behavior:

- **Close-range silhouettes (0.1-0.5 m from lens):** Insects within 50 cm of the camera appear as large, partially defocused shapes with visible wing outline, body segmentation, and antenna geometry. These close-range events are the highest-value classification targets because morphological features are resolvable at standard 1080p/2K sensor resolution. A moth with a 30 mm wingspan at 20 cm produces an image footprint of approximately 200×150 pixels on a 1080p sensor with a 130° FOV lens.
- **Mid-range flight tracks (0.5-3 m):** Insects in the mid-field appear as compact bright spots (3-30 pixels) moving against a static background. Flight trajectory kinematics (speed, turning rate, hovering duration, approach/departure angle) provide species-level discriminative features even when morphological detail is not resolvable. Mosquitoes hover at 1.5-2.5 m/s with 30-60 Hz wingbeat flicker; moths exhibit erratic spiraling approach at 0.5-1.5 m/s; beetles fly in straight lines at 2-4 m/s.
- **Landing and resting behavior (on camera housing):** Many insects land directly on or near the camera enclosure, remaining stationary for seconds to minutes. These resting specimens present dorsal or lateral views at near-macro distances (5-20 mm from the lens outer element), providing extremely high-resolution morphological data. Body length, segment proportions, wing venation patterns, and leg structure are all resolvable.

### 2. Image Processing Pipeline

The system processes camera frames during NIR night-mode operation using a four-stage pipeline executed on the camera's application processor (typically ARM Cortex-A53/A55 or equivalent):

**Stage 1: Temporal background model.** A running Gaussian mixture model (3 components per pixel, learning rate α = 0.005) maintains the static scene background. Foreground pixels are those exceeding 3σ from all mixture components. This isolates moving or newly-appeared objects (insects, but also rain, spider webs, windblown debris) from the static background (walls, plants, ground). The background model adapts to slow illumination changes (moon phase, passing car headlights) while preserving sensitivity to small, fast-moving insects.

**Stage 2: Motion classification.** Foreground regions are tracked across consecutive frames using a Kalman filter bank (max 50 simultaneous tracks). Each track is classified as insect-candidate, spider web (oscillatory motion anchored to fixed points), rain (high-velocity linear trajectories with consistent direction across all tracks), or debris (single-event ballistic trajectory). Only insect-candidate tracks proceed to Stage 3.

**Stage 3: Morphological feature extraction.** For each insect-candidate track, the system extracts features from the frame with highest spatial resolution (closest approach to camera). Features include: body length (major axis of fitted ellipse, calibrated to physical units using estimated distance from focus quality), aspect ratio (major/minor axis), wing presence and extension state (detected via bilateral symmetry analysis of the silhouette), antenna geometry (length relative to body, branching pattern if resolvable), leg count visibility (ventral view only), and wingbeat frequency (extracted from temporal intensity modulation of the insect's image across consecutive frames at 15-30 fps capture rate).

**Stage 4: Species classification.** A MobileNetV3-Small backbone (width multiplier 0.5, input resolution 96×96, quantized to INT8) processes a cropped, normalized image patch of each insect detection. The classifier outputs probabilities over 47 target classes spanning 12 pest-relevant insect families:

| Family | Target genera/species | Pest relevance |
|--------|----------------------|----------------|
| Rhinotermitidae | *Reticulitermes* spp. (alates) | Subterranean termite swarmers |
| Kalotermitidae | *Incisitermes*, *Cryptotermes* (alates) | Drywood termite swarmers |
| Formicidae | *Camponotus* spp. (alates) | Carpenter ant reproductive swarms |
| Culicidae | *Aedes*, *Culex*, *Anopheles* | Disease vectors (WNV, EEE, Zika, malaria) |
| Pyralidae | *Plodia interpunctella*, *Ephestia* spp. | Pantry moths, stored product pests |
| Scolytinae | *Dendroctonus*, *Ips* spp. | Bark beetles (tree mortality) |
| Cerambycidae | *Anoplophora*, *Monochamus* | Longhorn borers (structural/tree damage) |
| Blattidae | *Periplaneta americana* | American cockroach (sanitation) |
| Muscidae | *Musca domestica*, *Stomoxys* | Filth flies (sanitation, livestock) |
| Psychodidae | *Clogmia albipunctata* | Drain flies (plumbing indicator) |
| Lampyridae | *Photinus*, *Photuris* spp. | Fireflies (biodiversity indicator, non-pest) |
| Other | Non-target Lepidoptera, Ephemeroptera, etc. | Background species for density normalization |

The classifier is trained on a composite dataset assembled from: the Global Biodiversity Information Facility (GBIF, 3.2M insect occurrence images), the iNaturalist open dataset (12M insect observations with community-verified taxonomic labels), the Copenhagen insect dataset (time-lapse camera trap images of insects at light sources), and synthetic training examples generated by capturing insects on controlled NIR-illuminated backgrounds matching typical doorbell camera imaging conditions.

### 3. Population Density Estimation

Raw detection counts per species per night are converted to calibrated population density estimates through three correction factors:

**Phototaxis correction factor (PCF):** Different species exhibit vastly different attraction strengths to 850/940 nm NIR. Owens et al. (2020) quantified relative spectral attraction coefficients for major insect orders. The system applies order-specific PCFs to normalize raw counts: Lepidoptera PCF = 1.0 (reference), Coleoptera PCF = 0.4 (weaker NIR attraction), Diptera PCF = 0.7, Isoptera PCF = 1.3 (strong attraction during swarming).

**Environmental normalization:** Nightly counts are normalized against temperature (insect flight activity drops below 10°C and above 38°C, with species-specific optima), wind speed (sustained winds above 15 km/h suppress small insect flight), moon phase (full moon reduces phototactic attraction to artificial sources by 30-60% for many species), and precipitation (rain suppresses flight activity entirely; rain detection from Stage 2 motion classification provides binary rain/no-rain gating).

**Camera hardware normalization:** Different camera models have different NIR LED power, spectral characteristics, and sensor sensitivity. The system maintains a camera-model database mapping each device's hardware profile to a normalization coefficient. A Ring Video Doorbell Pro 2 (850 nm, 1,200 mW radiant flux) attracts approximately 2.3× the insects of a Wyze Cam v3 (940 nm, 400 mW), independent of species composition.

### 4. Temporal Trend Analysis and Alert Generation

The system maintains per-species nightly count time series and applies three detection algorithms:

**Swarming event detection:** Termite and carpenter ant reproductive flights produce dramatic single-night spikes: a mature Reticulitermes colony releases 100-1,000 alates over 30-90 minutes, typically triggered by afternoon rainfall followed by warm evening temperatures. The system detects swarming events as single-night counts exceeding 10× the 30-day rolling median for the relevant taxon, with temporal concentration (>80% of detections within a 2-hour window) confirming coordinated reproductive flight rather than gradual population increase.

**Seasonal phenology shift detection:** Climate change is shifting insect emergence timing by 2-5 days per decade for many temperate species. The system records first-of-season detection dates for each species and compares against historical baselines (built from 3+ years of local camera data). Statistically significant advancement of spring emergence dates (p < 0.05, one-sided t-test against baseline) triggers informational alerts and contributes to regional phenology tracking databases.

**Infestation onset detection:** A sustained increase in species-specific counts across multiple consecutive nights, modeled as a change-point in a Bayesian online change-point detection framework (Adams & MacKay, 2007), indicates a growing local population rather than transient foraging. The system distinguishes: a new colony establishing nearby (monotonic increase over 2-4 weeks), seasonal population peak (bell-shaped curve matching historical phenology), and migration/dispersal event (elevated counts for 1-3 nights followed by return to baseline). Only the first pattern triggers a pest alert.

### 5. Neighborhood-Scale Spatial Correlation

When multiple cameras within a neighborhood participate, spatial correlation algorithms identify:

**Breeding hotspot localization:** If cameras on three sides of a city block detect elevated Culex mosquito counts while the fourth side shows baseline levels, the breeding source is likely in the high-count direction. A 2D Gaussian plume dispersion model fitted to the spatial count gradient estimates the probable location of standing water or other breeding habitat within 50-100 meters.

**Dispersal corridor mapping:** Sequential detection of swarming alates across cameras on successive nights, with the wavefront propagating at 100-500 meters per night (consistent with measured subterranean termite dispersal range of 200-400 m from parent colony during nuptial flights, Nutting 1969), maps the dispersal front of a reproductive swarm.

**Treatment efficacy verification:** After pest treatment, the camera system monitors whether target species counts decline to baseline while neighboring cameras show no corresponding decline. Count rebound within 2-4 weeks indicates treatment failure and triggers re-application alerts.

### 6. Privacy and Data Handling

The system processes all imagery on-device. No raw video frames leave the camera. Only structured metadata (species label, confidence score, timestamp, count, morphological feature vector) is transmitted to the cloud aggregation layer. The spatial correlation features use differential privacy (ε = 1.0) when sharing count data between neighboring cameras, adding calibrated Laplace noise to individual camera counts before spatial aggregation.

## Claims

1. A system for automated pest insect surveillance at residential properties, comprising: a consumer smart doorbell or security camera with near-infrared illumination operating in night mode; an image processing pipeline running on the camera's edge processor that applies temporal background subtraction and motion tracking to isolate insect detections from static scene elements; and a convolutional neural network classifier that identifies detected insects to the family or genus level based on morphological features and flight trajectory kinematics extracted from NIR image frames.

2. The system of claim 1, wherein the classifier targets pest-relevant insect families including Rhinotermitidae, Kalotermitidae, Formicidae, Culicidae, Pyralidae, Scolytinae, Cerambycidae, and Blattidae, distinguishing pest species from non-target background insects.

3. The system of claim 1, wherein morphological features extracted for classification include body length, body aspect ratio, wing venation pattern, bilateral wing symmetry, antenna geometry, and wingbeat frequency estimated from temporal intensity modulation across consecutive video frames.

4. The system of claim 1, further comprising a population density estimation module that corrects raw detection counts using species-specific phototaxis correction factors, environmental normalization (temperature, wind speed, moon phase, precipitation), and camera hardware normalization based on NIR LED power and spectral characteristics.

5. The system of claim 1, further comprising a swarming event detection module that identifies reproductive flight events from single-night count spikes exceeding a configurable multiple of the rolling median for the relevant taxon, with temporal concentration confirming coordinated swarm behavior.

6. The system of claim 1, further comprising a Bayesian online change-point detection module that distinguishes infestation onset (monotonic increase over 2-4 weeks) from seasonal population peaks (bell-shaped phenological curve) and transient migration events (1-3 night elevation followed by return to baseline).

7. A method for neighborhood-scale pest population monitoring comprising: collecting nightly species-specific insect counts from multiple participating smart cameras within a geographic area; applying differential privacy noise to individual camera counts; aggregating counts across cameras to generate spatial density gradient maps; and fitting dispersal models to the spatial gradient to estimate probable breeding site locations and dispersal corridor directions.

8. The method of claim 7, further comprising treatment efficacy verification that monitors whether target species counts at a treated property decline to baseline while neighboring camera counts remain stable, and generates re-treatment alerts when counts rebound within a configurable monitoring window.

9. The system of claim 1, wherein phenological trend analysis compares first-of-season detection dates for each species against historical baselines built from 3 or more years of local camera data, detecting statistically significant shifts in emergence timing attributable to climate variation.

10. The system of claim 1, wherein all image processing and classification occurs on the camera's edge processor, with no raw video frames transmitted from the device, and only structured metadata shared with a cloud aggregation layer.

11. The system of claim 1, wherein flight trajectory kinematics including speed, turning rate, hovering duration, and approach angle provide species-level discrimination for insects too distant for morphological feature resolution, with distinct kinematic signatures for Culicidae (hovering, 1.5-2.5 m/s), Lepidoptera (erratic spiraling, 0.5-1.5 m/s), and Coleoptera (linear flight, 2-4 m/s).

## Prior Art References

1. National Pest Management Association — $5B annual US termite damage estimate
2. CDC Vector-Borne Disease Division — US mosquito-borne disease surveillance data
3. Rust & Su, Annual Review of Entomology, 2012 — Termite inspection detection rates
4. Mankin et al., Florida Entomologist, 2002 — Acoustic detection of termite feeding
5. Owens et al., Philosophical Transactions B, 2020 — Insect phototaxis review (229 studies)
6. van Grunsven et al., Insect Conservation and Diversity, 2014 — Automated light trap vs. professional survey correlation
7. Global Biodiversity Information Facility (GBIF) — 3.2M insect occurrence images
8. iNaturalist — 12M community-verified insect observations
9. Adams & MacKay, 2007 — Bayesian Online Changepoint Detection
10. Nutting, 1969 — Termite nuptial flight dispersal ranges
11. Statista, 2025 — Smart home security camera market data
12. MobileNetV3, Howard et al., 2019 — Efficient mobile CNN architecture
13. TensorFlow Lite for Microcontrollers — Edge ML deployment framework

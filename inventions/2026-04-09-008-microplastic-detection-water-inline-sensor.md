# System and Method for Continuous Microplastic Detection and Polymer-Type Classification in Municipal Water Distribution Systems Using Inline Multi-Angle Light Scattering and On-Device Neural Network Particle Identification

**LITF-PA-2026-008 · WaterTech / Environmental AI**
**Published:** 2026-04-09
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is an inline flow-through optical sensor system for continuous, real-time detection and classification of microplastic particles in municipal water distribution systems. The system uses a 635 nm diode laser illuminating a hydrodynamically focused sample stream within a flow-through optical cell, with an array of eight silicon photodetectors positioned at angles from 5° to 170° to capture the multi-angle light scattering (MALS) profile of each particle as it transits the detection volume. A 1D convolutional neural network (1D-CNN) running on an embedded ARM Cortex-M7 microcontroller classifies each particle into polymer type (polyethylene, polypropylene, polystyrene, polyethylene terephthalate, nylon, and non-plastic mineral/organic), estimates particle size (1-500 μm), and counts particle flux per unit volume. The sensor node costs under $120 in bill-of-materials, operates continuously at flow rates up to 2 L/min with a detection limit of 0.1 particles/L for particles ≥10 μm, and communicates via LoRaWAN to a cloud dashboard for network-wide microplastic mapping. A distributed deployment across a water distribution network enables identification of contamination sources, pipe material degradation patterns, and treatment effectiveness monitoring.

## Field of the Invention

This invention relates to environmental water quality monitoring, specifically to inline optical sensing and machine learning classification of microplastic particles in pressurized water distribution systems.

## Background

Microplastic contamination of drinking water is a rapidly emerging public health concern. The [World Health Organization (2019)](https://www.who.int/publications/i/item/9789241516198) published a comprehensive review finding microplastics present in drinking water globally, though noting insufficient data on health effects. [Kosuth et al. (2018)](https://doi.org/10.1371/journal.pone.0194970) found an average of 5.45 microplastic particles per liter in tap water samples across 14 countries. [Mintenig et al. (2019)](https://doi.org/10.1016/j.watres.2019.04.015) demonstrated that conventional water treatment plants remove 70-80% of microplastics but a significant fraction (particularly particles <20 μm) passes through to the distribution system.

Current microplastic detection methods are entirely laboratory-based:

- **FTIR microscopy:** Fourier-transform infrared spectroscopy coupled with optical microscopy is the gold standard for microplastic identification. Cost: $50,000-$150,000 per instrument. Processing time: 4-24 hours per sample. Requires sample collection, filtration, and transport to a laboratory. See [Löder & Gerdts (2015)](https://doi.org/10.1007/978-3-319-16510-3_8) for methodology.
- **Raman microscopy:** Confocal Raman spectroscopy provides polymer identification at higher spatial resolution than FTIR. Cost: $100,000-$300,000. Processing time: 8-48 hours per sample. See [Araujo et al. (2018)](https://doi.org/10.1016/j.scitotenv.2018.01.047).
- **Nile Red fluorescence:** [Maes et al. (2017)](https://doi.org/10.1021/acs.analchem.7b00112) demonstrated Nile Red staining as a rapid screening method. Provides total plastic count but cannot distinguish polymer types. Not suitable for inline deployment due to chemical addition requirements.
- **Flow cytometry:** Commercial flow cytometers (Beckman Coulter, BD Biosciences) can detect particles by size and fluorescence but are not designed for polymer identification and cost $50,000-$200,000.

Relevant patents include:

- [US20210181087A1](https://patents.google.com/patent/US20210181087A1) (Paterson): Describes a method for detecting microplastics using hyperspectral imaging of filtered water samples. Requires sample collection and filtration; not inline or real-time.
- [US11313786B2](https://patents.google.com/patent/US11313786B2) (Horiba): Multi-angle light scattering instrument for particle characterization in laboratory settings. Bench-top instrument costing >$30,000; not designed for embedded deployment or microplastic classification.
- [US20220057317A1](https://patents.google.com/patent/US20220057317A1) (Purdue University): Machine learning classification of microplastics from FTIR spectra. Post-hoc analysis of laboratory data; no inline sensing component.

The gap in the art is: (a) an inline sensor that operates continuously in a pressurized water pipe without sample collection, (b) real-time polymer-type classification without spectroscopic instruments, (c) a cost point (<$200) enabling distributed deployment across a water network, and (d) a system that maps microplastic contamination spatially across a distribution network to identify sources and trends.

## Detailed Description

### 1. Optical Cell Design

The flow-through optical cell is a precision-machined aluminum housing (dimensions: 80 × 40 × 30 mm) with fused silica windows at the laser entry and eight detector positions. Key design parameters:

- **Flow channel:** 500 μm × 500 μm square cross-section at the detection point, expanding to 4 mm diameter at inlet and outlet ports. Hydrodynamic focusing is achieved by a sheath flow arrangement where filtered water (0.2 μm filter) surrounds the sample stream, constraining particles to the central 100 μm of the channel. This ensures single-file particle transit through the laser beam.
- **Laser source:** 635 nm diode laser (5 mW, TEM00 mode), focused to a 50 μm beam waist at the detection volume using a single aspheric lens. 635 nm is chosen to balance scattering intensity (increases with shorter wavelength) against water absorption (increases below 600 nm for dissolved organics).
- **Detector array:** Eight Hamamatsu S1227-1010BQ silicon photodiodes (active area: 10 mm²) positioned at 5°, 15°, 30°, 60°, 90°, 120°, 150°, and 170° relative to the forward beam direction. Each detector has a 1 mm aperture to define angular acceptance (±2°). Detectors are read by a simultaneous-sampling 16-bit ADC at 500 kSPS per channel.
- **Trigger mechanism:** A forward-scatter detector at 5° triggers particle detection when the signal exceeds a threshold corresponding to a 5 μm polystyrene sphere equivalent. The trigger initiates a 200 μs acquisition window capturing the complete scattering profile as the particle transits the beam.

### 2. Mie Scattering Physics and Feature Extraction

When a particle transits the laser beam, the angular distribution of scattered light is governed by [Mie theory](https://doi.org/10.1002/andp.19083300302), with the pattern depending on particle size (relative to wavelength), shape, and complex refractive index. The key physical principle enabling polymer classification is that different polymers have distinct refractive indices at 635 nm:

| Polymer | Refractive Index (635 nm) |
|---------|--------------------------|
| Polyethylene (PE) | 1.50 |
| Polypropylene (PP) | 1.49 |
| Polystyrene (PS) | 1.59 |
| PET | 1.58 |
| Nylon (PA6) | 1.53 |
| Mineral (calcite) | 1.66 |
| Organic (cellulose) | 1.47 |

For particles in the 10-500 μm range (size parameter x = πd/λ = 50-2500), the angular scattering pattern has sufficient structure to differentiate these refractive indices. The 8-angle measurement captures the key features: forward lobe intensity (dominated by size), side-scatter ratio (sensitive to refractive index), and backscatter intensity (sensitive to surface roughness and internal structure).

From each particle's 8-angle scattering profile, the system extracts a 24-dimensional feature vector:

1. **8 raw intensities:** Normalized by the forward-scatter (5°) intensity
2. **8 asymmetry ratios:** Ratios of complementary angle pairs (5°/170°, 15°/150°, 30°/120°, 60°/90°)
3. **4 spectral moments:** Mean, variance, skewness, and kurtosis of the angular intensity distribution
4. **2 shape parameters:** Degree of linear polarization (if a polarizer is added to 2 detectors) and depolarization ratio indicating non-sphericity
5. **2 temporal features:** Transit time (correlated with flow velocity and particle size) and pulse shape (Gaussian for spheres, non-Gaussian for fibers)

### 3. On-Device 1D-CNN Architecture

A 1D convolutional neural network processes the 24-dimensional feature vector to produce classification and sizing outputs:

- **Input:** 24-element vector
- **Conv1D block 1:** 32 filters, kernel size 3, ReLU activation, batch normalization
- **Conv1D block 2:** 64 filters, kernel size 3, ReLU activation, batch normalization, max pooling (2)
- **Dense layer 1:** 128 neurons, ReLU, dropout 0.3
- **Output heads:**
  - Polymer classification: 7-class softmax (PE, PP, PS, PET, nylon, mineral, organic)
  - Size estimation: Single neuron with ReLU (continuous μm output)
  - Confidence: Single neuron with sigmoid

Total parameters: approximately 18,000. Model size after INT8 quantization: 22 KB. Inference time: <1 ms on ARM Cortex-M7 at 480 MHz (STM32H7 series).

Training data is generated from: (a) Mie theory simulations covering the full parameter space (polymer type × size × shape × orientation, 500,000 synthetic samples with realistic noise), (b) laboratory validation measurements using reference microplastic particles ([Cospheric monodisperse microspheres](https://www.cospheric.com/) in PE, PS, and PMMA) and environmental samples characterized by FTIR as ground truth.

### 4. Network-Level Deployment and Source Identification

Multiple sensor nodes deployed across a water distribution network create a spatial map of microplastic contamination. Recommended deployment points:

- **Treatment plant outlet:** Establishes baseline contamination level after treatment
- **Distribution storage tanks:** Inlet and outlet monitoring to detect accumulation or resuspension
- **Service area boundaries:** 1-2 sensors per distribution pressure zone
- **Critical facilities:** Schools, hospitals, dialysis centers

Each node communicates particle counts and polymer-type distributions every 15 minutes via LoRaWAN to a cloud platform. The platform computes:

- **Spatial gradient analysis:** If downstream nodes show higher contamination than upstream nodes, the intervening pipe segments are flagged as potential sources (pipe degradation, service line leaching, cross-connection contamination).
- **Temporal trend analysis:** Long-term increases in specific polymer types (e.g., PVC particles increasing may indicate aging PVC pipe degradation).
- **Treatment effectiveness:** Comparison of pre/post-treatment levels after filter replacements or process changes.
- **Regulatory reporting:** Automated generation of microplastic monitoring reports as anticipatory compliance with emerging regulations (California [SB 1263](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=201720180SB1263) requires microplastic monitoring in drinking water).

### 5. Calibration and Quality Assurance

Each sensor node includes an automated calibration system:

- **Size calibration:** 20 μm NIST-traceable polystyrene microspheres injected from a sealed ampoule every 24 hours. The measured scattering pattern is compared against the known Mie solution, and the detector gain factors are adjusted.
- **Background subtraction:** A solenoid valve periodically diverts sample flow through a 0.2 μm filter, establishing the particle-free scattering background (stray light, window contamination). This background is subtracted from subsequent measurements.
- **Fouling detection:** Forward-scatter baseline intensity is monitored continuously. A sustained decrease exceeding 20% triggers a cleaning cycle (high-velocity flush) and a maintenance alert if cleaning does not restore baseline.

### 6. Figures Description

- **Figure 1:** Exploded view of the optical cell showing laser entry port, hydrodynamic focusing channel, detection volume, and 8-position detector ring with angular designations.
- **Figure 2:** Simulated Mie scattering polar plots for 50 μm spheres of PE (n=1.50), PS (n=1.59), and calcite (n=1.66) at 635 nm, showing discriminable angular intensity patterns.
- **Figure 3:** 1D-CNN architecture diagram from 24-element input through convolutional layers to dual-head output (classification + sizing).
- **Figure 4:** Network deployment map showing sensor node positions in an example distribution system, with spatial contamination gradient visualization and pipe segment source identification.

## Claims

- A method for real-time detection and classification of microplastic particles in water, comprising: illuminating a hydrodynamically focused sample stream with a laser source within an inline flow-through optical cell; measuring scattered light intensity at a plurality of angles using a photodetector array; extracting a feature vector from said multi-angle scattering measurements for each detected particle; and classifying each particle into a polymer type using an on-device neural network.

- The method of claim 1, wherein said photodetector array comprises at least eight detectors positioned at angles spanning from 5° to 170° relative to the forward beam direction.

- The method of claim 1, wherein said neural network is a 1D convolutional neural network quantized to INT8 representation and executing on an embedded microcontroller with inference latency less than 1 millisecond per particle.

- The method of claim 1, wherein said polymer type classification distinguishes at least polyethylene, polypropylene, polystyrene, polyethylene terephthalate, nylon, and non-plastic particles based on differences in complex refractive index at the laser wavelength.

- The method of claim 1, further comprising estimating particle size from the forward-scatter intensity and angular scattering pattern.

- A system for continuous inline microplastic monitoring comprising: a flow-through optical cell with hydrodynamic focusing; a laser source; a multi-angle photodetector array; an embedded processor executing a trained neural network for particle classification and sizing; and a wireless communication module for transmitting particle count and classification data to a central monitoring platform.

- The system of claim 6, wherein said system is deployed at a plurality of points across a water distribution network, and a central platform computes spatial contamination gradients to identify pipe segments that are sources of microplastic particles.

- The system of claim 6, further comprising an automated calibration subsystem that periodically measures reference microspheres and particle-free background to maintain measurement accuracy.

- The system of claim 6, wherein the bill-of-materials cost is less than $200 per sensor node.

- The method of claim 1, wherein said feature vector comprises raw angle-normalized intensities, asymmetry ratios between complementary angle pairs, spectral moments of the angular distribution, and temporal transit features.

## Prior Art References

- [WHO Microplastics in Drinking-Water (2019)](https://www.who.int/publications/i/item/9789241516198) — Global review of microplastics in drinking water
- [Kosuth et al.](https://doi.org/10.1371/journal.pone.0194970) — "Anthropogenic contamination of tap water" — PLOS ONE 2018
- [Mintenig et al.](https://doi.org/10.1016/j.watres.2019.04.015) — "Microplastic removal in drinking water treatment" — Water Research 2019
- [Löder & Gerdts](https://doi.org/10.1007/978-3-319-16510-3_8) — "FTIR Methodology for Microplastics" — Marine Anthropogenic Litter 2015
- [Araujo et al.](https://doi.org/10.1016/j.scitotenv.2018.01.047) — "Raman Spectroscopy for Microplastics" — Science of Total Environment 2018
- [Maes et al.](https://doi.org/10.1021/acs.analchem.7b00112) — "Nile Red Staining for Rapid Microplastic Detection" — Analytical Chemistry 2017
- [US20210181087A1](https://patents.google.com/patent/US20210181087A1) — Paterson — "Hyperspectral Microplastic Detection" (2021)
- [US11313786B2](https://patents.google.com/patent/US11313786B2) — Horiba — "Multi-Angle Light Scattering Instrument" (2022)
- [US20220057317A1](https://patents.google.com/patent/US20220057317A1) — Purdue — "ML Classification of Microplastics from FTIR" (2022)
- [Mie, G.](https://doi.org/10.1002/andp.19083300302) — "Beiträge zur Optik trüber Medien" — Annalen der Physik 1908
- [California SB 1263](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=201720180SB1263) — Microplastic monitoring mandate for drinking water
- [Cospheric Microspheres](https://www.cospheric.com/) — Reference microplastic particles for calibration
- [STM32H7 Series](https://www.st.com/en/microcontrollers-microprocessors/stm32h7-series.html) — ARM Cortex-M7 embedded platform
- [Semtech LoRa](https://www.semtech.com/lora) — LoRaWAN low-power wide-area network for IoT sensor communication

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*

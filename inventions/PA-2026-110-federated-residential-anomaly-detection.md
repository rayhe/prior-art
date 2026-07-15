# System and Method for Federated Anomaly Detection Across Heterogeneous Residential Security Camera Networks Using Privacy-Preserving Behavioral Embedding Extraction and Cross-Property Spatiotemporal Correlation

**LITF-PA-2026-110 · Residential Security / Federated Learning / Privacy Engineering**
**Published:** 2026-07-15
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting anomalous human behavior patterns that span multiple residential properties within a neighborhood, without any participating property owner sharing raw video footage with other participants. Each property runs an edge inference pipeline on its existing consumer security cameras (Ring, Arlo, Nest, UniFi Protect, Reolink, or similar) to extract fixed-length behavioral embedding vectors that encode locomotion trajectory shape, dwell-time distribution, approach-angle sequence, time-of-day profile, and gait periodicity for each detected human subject. These embeddings are stripped of biometric identifiers through a learned privacy projection that maps facial features and body proportions to a null subspace while preserving behavioral discriminability. Embeddings are shared over an encrypted peer-to-peer mesh network to a neighborhood-level graph neural network (GNN) aggregator that models inter-property spatiotemporal relationships. The GNN detects cross-property anomalies that no single camera system could identify alone: reconnaissance patterns where a subject visits multiple properties in sequence, recurrence of behaviorally similar subjects across days, and coordinated multi-actor surveillance of a target property from different neighboring vantage points.

## Field of the Invention

This invention relates to distributed video surveillance analytics, specifically to privacy-preserving methods for correlating human behavioral patterns across independently owned and operated residential security camera systems to detect neighborhood-level security threats.

## Background

Residential security cameras are now deployed in an estimated [53% of U.S. households](https://www.safehome.org/home-security-statistics/) (SafeHome.org, 2024), with the installed base growing at approximately 20% annually. Major consumer camera ecosystems include Amazon Ring (with over [10 million active Neighbors app users](https://ring.com/neighbors)), Google Nest, Arlo, Wyze, UniFi Protect, and Reolink. Each system operates as a silo: video is stored locally or in a vendor-specific cloud, and analysis is limited to events within a single property's camera coverage.

This siloed architecture creates a fundamental detection gap. Property crime, particularly residential burglary, frequently involves reconnaissance across multiple target properties. [Wright and Decker (1994)](https://www.ojp.gov/pdffiles1/nij/grants/248384.pdf) documented that 87% of convicted burglars reported surveilling target properties before committing the offense. [Bureau of Justice Statistics data](https://www.bjs.gov/content/pub/pdf/vdhb.pdf) shows that 65% of residential burglaries occur in neighborhoods where the offender has been previously observed. A subject who walks slowly past three houses on the same block, pausing at each driveway, generates three separate low-priority "person detected" events on three separate camera systems. No individual system has enough context to recognize the pattern as casing behavior.

Existing approaches to cross-camera correlation require centralized video access:

- **Cloud-based re-identification:** [US20230162580A1](https://patents.google.com/patent/US20230162580A1) (Verkada) describes cross-camera person re-identification using trajectory graphs, but operates within a single organization's camera network with centralized video access.
- **Law enforcement fusion centers:** Systems like [Axon Fusus](https://www.axon.com/products/fusus) aggregate camera feeds from participating businesses and residents, but require all participants to share raw video with a central authority, creating privacy, liability, and Fourth Amendment concerns.
- **Ring Neighbors / Citizen / Nextdoor:** Social platforms that rely on residents manually posting clips after observing suspicious activity. Detection depends entirely on human vigilance, introduces hours of latency, and sharing video publicly raises privacy concerns for innocent passersby.
- **Federated learning for surveillance:** [Ullah et al., Sensors 2023](https://doi.org/10.3390/s23249873) survey federated learning for video surveillance but focus on training shared classifiers across institutional camera networks, not on detecting behavioral anomalies that span independently owned residential systems.

The gap in the art is a system that: (a) detects behavioral anomalies spanning multiple independently owned residential properties; (b) operates without sharing raw video, biometric data, or personally identifiable information between property owners; (c) works across heterogeneous camera hardware and vendor ecosystems; (d) runs edge inference locally at each property without requiring cloud processing; and (e) provides real-time alerts for patterns that no single property's cameras could identify alone.

## Detailed Description

### 1. System Architecture Overview

The system comprises three tiers: (a) per-property edge nodes running behavioral embedding extraction on existing camera hardware or a dedicated compute module (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi 5 with Coral TPU, or Intel NCS2); (b) an encrypted peer-to-peer mesh network for embedding exchange between participating properties; and (c) a neighborhood-level GNN aggregator running on one or more designated nodes that models inter-property behavioral correlations and generates cross-property alerts.

Each participating property retains full ownership and control of its raw video. The system is explicitly designed so that no raw video frame, face encoding, or biometrically identifiable feature ever leaves the property's local compute node. Only privacy-projected behavioral embeddings are shared.

### 2. Per-Property Edge Inference Pipeline

The edge pipeline processes video from each camera through the following stages:

**2.1 Human Detection and Tracking:** A lightweight object detector (e.g., YOLOv8-nano, 3.2M parameters, ~15 ms inference on Jetson Orin Nano at 640×640) identifies human subjects in each frame. A multi-object tracker (ByteTrack or BoT-SORT) maintains subject identity across frames within a single camera's field of view, assigning a local track ID that persists only within that camera session.

**2.2 Cross-Camera Track Association (Intra-Property):** For properties with multiple cameras, an intra-property re-identification module associates tracks across cameras using appearance features (OSNet-x0.25, 0.2M parameters) combined with spatiotemporal transition probability models learned from the property's camera topology. This association is performed entirely on-device and produces a unified per-subject trajectory across all property cameras.

**2.3 Behavioral Feature Extraction:** For each completed subject trajectory (defined as the interval from first detection to 60 seconds after last detection), the system computes a behavioral feature vector comprising:

- **Trajectory shape features (12 dimensions):** Fourier descriptor coefficients (first 6 harmonics) of the subject's 2D trajectory projected onto the property's ground plane via homography. These encode the geometric shape of movement (straight pass-through vs. meandering vs. U-turn vs. approach-and-retreat) independent of absolute position or scale.
- **Dwell-time distribution (8 dimensions):** Histogram of time spent in each of 8 angular sectors radiating from the property's centroid, normalized by total visit duration. A subject who lingers at the driveway and garage side but ignores the front door produces a distinctive distribution different from a delivery driver or solicitor.
- **Approach-retreat kinematic profile (6 dimensions):** Mean and variance of velocity during approach phase, on-property phase, and departure phase. Subjects who slow down approaching, spend extended time near structures, and accelerate departing exhibit a profile distinct from casual pedestrians.
- **Time-of-day encoding (4 dimensions):** Sinusoidal encoding of visit time (sin/cos of hour-of-day and day-of-week), capturing temporal periodicity without revealing absolute timestamps.
- **Gait periodicity features (6 dimensions):** Autocorrelation peaks of vertical bounding-box oscillation during walking phases, encoding stride frequency and regularity. These features are not biometrically identifying at the resolution and noise levels of consumer security cameras but do enable same-session behavioral consistency checks.
- **Interaction event flags (8 dimensions):** Binary or soft indicators for: approached front door, approached garage, examined windows, looked over fence, photographed property, carried tools/bag, accompanied by vehicle, accompanied by other persons.

The full behavioral feature vector is 44 dimensions before privacy projection.

### 3. Privacy-Preserving Embedding Projection

The critical privacy guarantee is enforced by a learned linear projection that maps the 44-dimensional behavioral feature vector into a 32-dimensional embedding space while provably destroying biometric identifiability. The projection matrix W ∈ ℝ^(32×44) is trained via an adversarial objective:

- **Utility loss:** Minimize triplet loss over behavioral similarity (same behavioral pattern at different properties should produce nearby embeddings).
- **Privacy loss:** Maximize adversary's cross-entropy loss when attempting to recover face embeddings, height estimates, or body-proportion vectors from the projected embeddings. The adversary is a 3-layer MLP trained concurrently.
- **Regularization:** The projection matrix is constrained to have a condition number below 10 and a nuclear norm penalty to prevent degenerate mappings that could leak identity through outlier dimensions.

The privacy projection is trained offline on a synthetic dataset (generated by combining trajectory motion capture data with randomized anthropometric parameters) and distributed as a frozen model to all participating nodes. Formal privacy guarantees are provided via a differential privacy mechanism: Gaussian noise with σ calibrated to (ε=1.0, δ=10⁻⁵) is added to each embedding before transmission.

### 4. Peer-to-Peer Embedding Exchange

Participating properties form an encrypted mesh network using WireGuard tunnels over each property's existing internet connection. Each property advertises its geographic position (quantized to a 50-meter Uber H3 hex cell, resolution 10) so that the aggregator can model spatial relationships without learning exact addresses.

When a subject trajectory completes at Property A, the edge node computes the privacy-projected embedding and broadcasts it to the neighborhood mesh with the following metadata: originating H3 cell ID (not property address), trajectory start/end timestamps (quantized to 5-minute bins), trajectory duration (quantized to 30-second bins), a trajectory confidence score, and a cryptographic session nonce. No raw video, appearance features, or camera identifiers are included.

Embeddings are retained in the mesh for a configurable window (default: 72 hours) and then cryptographically erased via forward-secret key rotation.

### 5. Graph Neural Network Aggregator

The GNN aggregator maintains a dynamic heterogeneous graph where:

- **Property nodes** represent participating properties, with features encoding their H3 position and camera coverage characteristics (number of cameras, estimated coverage arc).
- **Embedding nodes** represent individual behavioral embeddings received from the mesh, with features encoding the 32-dimensional projected embedding plus temporal metadata.
- **Spatial edges** connect property nodes based on geographic adjacency (within 500 meters), weighted by walking-distance estimates from OpenStreetMap road network data.
- **Similarity edges** connect embedding nodes whose cosine similarity exceeds a learned threshold (initialized at 0.75), indicating potentially the same or behaviorally similar subjects observed at different properties.
- **Temporal edges** connect embedding nodes from the same H3 cell in chronological order, enabling recurrence detection.

The GNN architecture uses 3 layers of heterogeneous graph attention (HGT) with 64-dimensional hidden states, trained to classify subgraph patterns into anomaly categories.

### 6. Cross-Property Anomaly Detection

The GNN aggregator detects the following anomaly types, each defined by a characteristic subgraph pattern:

- **Sequential reconnaissance:** A cluster of high-similarity embeddings appearing at 3+ properties within a 2-hour window, with inter-property travel times consistent with pedestrian or slow-vehicle movement along the street network. Estimated false positive rate: < 0.5 per property per day at default threshold.
- **Recurrent unknown visitor:** High-similarity embeddings appearing at the same property on 3+ separate days within a 14-day window, where the behavioral profile does not match any property-owner-labeled "known person" template (mail carrier, neighbor, delivery driver). Each property owner maintains a local set of "known person" behavioral templates that are never shared.
- **Coordinated multi-actor surveillance:** Two or more distinct embedding clusters (different subjects) that appear at properties with overlapping sight lines to a common target property within a 30-minute window, with dwell-time distributions concentrated toward the target property.
- **Anomalous temporal-spatial pattern:** Embedding appearances at unusual times (e.g., 2-4 AM) at multiple properties, detected via seasonal decomposition of the neighborhood's aggregate temporal embedding density.

### 7. Alert Generation and Notification

When the GNN aggregator classifies a subgraph as anomalous with confidence exceeding a property-owner-configurable threshold (default: 0.85), it generates an alert containing: the anomaly type and confidence score; the set of affected H3 cells (not specific properties); the temporal window of the detected pattern; and a natural-language description generated from the subgraph features (e.g., "A behaviorally consistent subject was observed at 4 properties on Oak Street between 2:15 PM and 3:40 PM, with dwell times of 45-90 seconds at each property and approach patterns focused on side yards").

Alerts are delivered only to property owners whose properties are within the affected area. The alert never includes raw video, face images, or biometric data. Property owners may choose to review their own local camera footage for the relevant time window after receiving an alert.

### 8. Heterogeneous Camera Compatibility Layer

To support the diverse ecosystem of consumer security cameras, the edge inference pipeline includes a hardware abstraction layer that interfaces with camera systems via: RTSP streams (UniFi Protect, Reolink, Amcrest, Hikvision, and most IP cameras); ONVIF discovery and streaming protocol; vendor-specific local APIs (Ring's undocumented local streaming, Arlo's local storage mode, Nest/Dropcam local RTSP where available); and USB/CSI direct camera connections for dedicated edge hardware. The abstraction layer normalizes input to 640×480 at 5 FPS (sufficient for behavioral analysis while minimizing compute) and handles camera-specific quirks including fisheye dewarping, IR/visible switching, and variable frame rate compensation.

### 9. Figures Description

- **Figure 1:** System architecture showing per-property edge nodes with camera inputs, behavioral embedding extraction pipeline, privacy projection module, and encrypted mesh network connecting to a neighborhood-level GNN aggregator.
- **Figure 2:** Privacy projection training pipeline showing the adversarial architecture with utility and privacy loss branches, including the differential privacy noise injection stage.
- **Figure 3:** Example heterogeneous graph structure showing property nodes, embedding nodes, and the three edge types (spatial, similarity, temporal) with a highlighted sequential reconnaissance anomaly subgraph across four properties.
- **Figure 4:** Behavioral embedding feature extraction detail showing the six feature groups (trajectory shape, dwell distribution, approach-retreat kinematics, time encoding, gait periodicity, interaction events) computed from a sample subject trajectory across two cameras on a single property.
- **Figure 5:** Alert generation flow showing GNN subgraph classification, confidence thresholding, natural-language description generation, and privacy-filtered delivery to affected property owners.

## Claims

1. A system for detecting behavioral anomalies across multiple independently owned residential properties, comprising: a plurality of edge compute nodes, each associated with one or more consumer security cameras at a respective property, wherein each edge node extracts behavioral embedding vectors from detected human subject trajectories; a privacy projection module that maps behavioral embeddings into a reduced-dimension space while provably destroying biometric identifiability through an adversarial training objective; an encrypted peer-to-peer mesh network connecting participating edge nodes for embedding exchange; and a graph neural network aggregator that models inter-property spatiotemporal relationships between privacy-projected embeddings to detect cross-property anomaly patterns.

2. The system of claim 1, wherein the behavioral embedding vector comprises trajectory shape features encoded as Fourier descriptor coefficients of the subject's ground-plane projected path, dwell-time angular distribution relative to the property centroid, approach-retreat kinematic profiles, sinusoidal time-of-day encoding, gait periodicity features derived from bounding-box vertical oscillation autocorrelation, and interaction event indicators.

3. The system of claim 1, wherein the privacy projection module comprises a linear projection matrix trained with a joint loss function combining triplet loss for behavioral similarity preservation and adversarial loss for biometric non-recoverability, with differential privacy noise injection calibrated to (ε, δ)-guarantees before embedding transmission.

4. The system of claim 1, wherein the graph neural network aggregator maintains a dynamic heterogeneous graph with property nodes, embedding nodes, spatial adjacency edges weighted by walking distance, cosine-similarity edges between behaviorally similar embeddings, and temporal recurrence edges.

5. The system of claim 1, wherein cross-property anomaly detection includes sequential reconnaissance detection, defined as a cluster of high-similarity embeddings appearing at three or more properties within a configurable time window with inter-property travel times consistent with pedestrian or slow-vehicle movement along the street network.

6. The system of claim 1, wherein cross-property anomaly detection includes recurrent unknown visitor detection, defined as high-similarity embeddings appearing at the same property on three or more separate days within a configurable window, where the behavioral profile does not match any locally maintained known-person template.

7. The system of claim 1, wherein cross-property anomaly detection includes coordinated multi-actor surveillance detection, defined as distinct embedding clusters from different subjects appearing at properties with overlapping sight lines to a common target property within a configurable time window.

8. A method for privacy-preserving neighborhood-level security monitoring comprising: extracting behavioral feature vectors from human subject trajectories detected by consumer security cameras at each participating property; applying a learned privacy projection that destroys biometric identifiability while preserving behavioral discriminability; transmitting privacy-projected embeddings over an encrypted mesh network with geographic metadata quantized to prevent exact property identification; aggregating embeddings into a heterogeneous graph modeling inter-property spatial, similarity, and temporal relationships; classifying subgraph patterns to detect cross-property anomalies including sequential reconnaissance, recurrent visitors, and coordinated surveillance; and delivering privacy-filtered alerts to affected property owners without including raw video or biometric data.

9. The method of claim 8, further comprising a hardware abstraction layer that normalizes video input from heterogeneous consumer camera systems including RTSP, ONVIF, and vendor-specific local APIs to a common format for behavioral feature extraction.

10. The system of claim 1, wherein embeddings transmitted over the mesh network include only: an originating geographic cell identifier quantized to a resolution preventing exact property identification, trajectory timestamps quantized to multi-minute bins, trajectory duration quantized to multi-second bins, and a trajectory confidence score, and explicitly exclude raw video frames, appearance features, camera identifiers, and property addresses.

11. The method of claim 8, wherein alerts generated from cross-property anomaly detection contain the anomaly type and confidence score, affected geographic cells, temporal window, and a natural-language description generated from subgraph features, and never contain raw video, face images, biometric data, or specific property addresses of other participating properties.

## Prior Art References

1. [SafeHome.org Home Security Statistics 2024](https://www.safehome.org/home-security-statistics/) — 53% U.S. household camera adoption
2. [Wright & Decker, Burglars on the Job (1994)](https://www.ojp.gov/pdffiles1/nij/grants/248384.pdf) — 87% of burglars conduct prior surveillance of target properties
3. [Bureau of Justice Statistics](https://www.bjs.gov/content/pub/pdf/vdhb.pdf) — Victimization during household burglary
4. [US20230162580A1](https://patents.google.com/patent/US20230162580A1) — Verkada — Cross-camera re-identification within single-organization networks
5. [Axon Fusus](https://www.axon.com/products/fusus) — Centralized law enforcement camera aggregation platform
6. [Ullah et al., Sensors 2023](https://doi.org/10.3390/s23249873) — Survey of federated learning for video surveillance
7. [ByteTrack (Zhang et al., ECCV 2022)](https://arxiv.org/abs/2209.02976) — Multi-object tracking by association
8. [OSNet (Zhou et al., ICCV 2019)](https://arxiv.org/abs/1905.00953) — Omni-Scale feature learning for person re-identification
9. [HGT — Heterogeneous Graph Transformer (Hu et al., WWW 2020)](https://arxiv.org/abs/2203.12235) — Graph attention for heterogeneous graph learning
10. [YOLOv8 (Ultralytics, 2023)](https://arxiv.org/abs/2304.00501) — Real-time object detection
11. [WireGuard](https://www.wireguard.com/) — Fast, modern VPN protocol for encrypted mesh networking
12. [Uber H3](https://h3geo.org/) — Hexagonal hierarchical spatial indexing system
13. [Dwork, Differential Privacy (2006)](https://dwork.seas.harvard.edu/differential-privacy) — Formal privacy guarantees for statistical data release

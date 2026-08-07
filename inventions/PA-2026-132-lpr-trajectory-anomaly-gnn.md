# PA-2026-132: System and Method for Real-Time Vehicle Trajectory Anomaly Detection and Behavioral Threat Scoring Using Distributed License Plate Reader Networks with Spatiotemporal Graph Neural Network Analysis

**Filing:** LITF-PA-2026-132  
**Domain:** Community Security / Computer Vision / Graph Neural Networks  
**Published:** August 7, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for detecting anomalous vehicle behavior within residential neighborhoods using a distributed network of license plate reader (LPR) cameras and spatiotemporal graph neural network (ST-GNN) analysis. The system continuously ingests plate-read events (plate hash, timestamp, camera ID, optional vehicle color and make/model from co-located classification models) from fixed LPR cameras deployed at entry points, intersections, and cul-de-sac terminals of a residential road network. A trajectory reconstruction module chains individual plate reads into multi-hop vehicle trajectories using a constrained shortest-path solver that respects the physical road graph, one-way restrictions, and speed-plausible transit times between cameras. A spatiotemporal graph neural network, where nodes represent camera locations and edges represent road segments with learned travel-time distributions, processes each trajectory as a temporal sequence of node activations and computes an anomaly score by comparing the observed trajectory embedding against a learned distribution of normal traffic patterns for that time-of-day and day-of-week. Specific anomaly detectors flag: repeated circuit patterns (vehicle traversing the same camera pair three or more times within a configurable window), dwell-time anomalies (elapsed time between sequential camera reads significantly exceeding the expected transit time, indicating the vehicle stopped or circled within the unmonitored segment), dead-end probing (vehicle entering and exiting multiple cul-de-sacs within a single visit), and temporal outliers (non-resident plates appearing during statistically unusual hours). A federated learning protocol enables multiple neighborhoods to collaboratively train the anomaly model without sharing raw plate data, using differential privacy guarantees on gradient updates. The system generates tiered alerts (advisory, elevated, actionable) delivered to community security operators or residents through a configurable notification pipeline.

## Field of the Invention

This invention relates to community security and intelligent surveillance systems, specifically to automated detection of suspicious vehicle behavior patterns using license plate reader data and graph-based machine learning operating on residential road network topology.

## Background

Property crime in residential neighborhoods frequently involves pre-operational surveillance by perpetrators. An FBI Uniform Crime Report analysis of convicted burglars found that 83% conducted at least one drive-through reconnaissance of the target neighborhood before committing the offense. Weisel (2002, ASU Center for Problem-Oriented Policing) documented that professional burglars typically visit a target area 2 to 5 times over 1 to 3 weeks, scouting entry/exit routes, noting resident schedules, and identifying unoccupied homes. Vehicle-based casing produces distinctive spatial signatures: repeated circuits on the same streets, slow traversal of residential blocks, U-turns at dead ends, and visits during unusual hours. These patterns are recognizable in aggregate but difficult for any single resident or camera operator to detect in real time.

License plate reader technology has matured rapidly for residential deployment. Fixed LPR cameras (e.g., Motorola/Vigilant, Flock Safety, community-operated systems using Ubiquiti AI cameras) capture plate numbers at accuracy rates exceeding 95% under normal conditions. As of 2025, over 80,000 Flock Safety cameras were deployed across 5,000+ communities in the United States. Residential LPR networks typically comprise 5 to 50 cameras covering neighborhood entry/exit points and key intersections.

Current LPR analytics are primarily reactive and plate-centric:

- **Hot-list matching:** Incoming plate reads are compared against databases of stolen vehicles, AMBER alerts, and law enforcement BOLOs. This catches known threats but is blind to unknown vehicles conducting pre-operational surveillance. EFF analysis found that 99.9% of plates scanned by ALPR systems are not on any hot list.
- **Frequency-based alerts:** Some systems alert when a non-resident plate appears more than N times within a configurable period. This produces high false-positive rates because delivery drivers, postal carriers, rideshare vehicles, and regular visitors generate repeat visits that are entirely benign. Flock Safety's "Frequent Flyer" feature uses simple count thresholds without considering the spatial or temporal structure of visits.
- **Geofence alerts:** Notifications when any plate enters or exits a defined zone. No trajectory analysis within the zone. No distinction between a vehicle driving straight through versus one that circles the block four times.

Graph neural networks (GNNs) have been applied to traffic forecasting and trajectory prediction in transportation research. Li et al. (ICLR 2018, "Diffusion Convolutional Recurrent Neural Network") demonstrated that modeling road networks as directed graphs and applying diffusion convolution captures spatial dependencies in traffic flow. Jiang and Luo (2022) applied spatiotemporal graph attention networks to vehicle trajectory prediction on highway networks. Zheng et al. (2020) introduced GMAN, a graph multi-attention network for traffic prediction that achieves state-of-the-art forecasting accuracy on urban road networks. None of these systems apply GNN-based anomaly detection to sparse, event-driven LPR data on residential road graphs for the purpose of security threat assessment.

The gap in the art is a system that: (a) reconstructs full vehicle trajectories from sparse LPR camera reads on a residential road graph, (b) applies spatiotemporal graph neural network analysis to score trajectory anomalousness against learned normal traffic patterns, (c) implements specific behavioral detectors for casing-associated patterns (circuiting, dead-end probing, dwell-time anomalies), (d) distinguishes between benign repeat visitors and genuinely anomalous behavior using trajectory structure rather than raw visit counts, and (e) enables federated model training across neighborhoods without sharing raw plate data.

## Detailed Description

### 1. LPR Network and Data Ingestion

The system operates on a network of fixed LPR cameras deployed at strategic positions within a residential neighborhood. Camera placement follows a coverage-maximization protocol targeting: neighborhood entry/exit points (every road connecting to arterials or adjacent developments), key T-intersections and four-way stops within the neighborhood, and cul-de-sac entrance points. A typical deployment of 15 to 40 cameras can achieve 85 to 95% coverage of vehicle movements within a neighborhood of 200 to 500 homes.

Each camera generates plate-read events comprising: a SHA-256 hash of the normalized plate string (uppercase, no spaces, standard character substitutions applied), a UTC timestamp with millisecond resolution, the camera's unique identifier with known GPS coordinates, a confidence score from the OCR engine (reads below 0.85 confidence are flagged for human review), and optional vehicle descriptor fields (color histogram, make/model classification from a co-located YOLO-v8-based vehicle classifier, direction-of-travel from sequential frame analysis). Events are ingested via a message queue with at-least-once delivery semantics. A deduplication stage collapses multiple reads of the same plate within a 5-second window at the same camera into a single event.

### 2. Road Network Graph Construction

The residential road network is modeled as a directed graph G = (V, E) where vertices V represent camera locations and edges E represent road segments connecting cameras. The graph is constructed by extracting street centerlines from OpenStreetMap, snapping cameras to nearest intersection nodes, computing shortest-path distances and expected travel times between all camera pairs via Dijkstra's algorithm, and pruning edges where the shortest-path distance exceeds 2 km.

Each edge carries attributes: physical road distance, speed limit-derived minimum transit time, learned mean transit time and standard deviation from observed data (updated hourly via exponential moving average), road type classification, and the number of unmonitored intersections between the camera pair.

### 3. Trajectory Reconstruction

Given a sequence of plate-read events for a single plate hash, the trajectory reconstruction module chains reads into coherent visit sessions. A visit session is initiated when a plate hash appears after an absence exceeding T_session (default: 4 hours). Within a session, consecutive reads are connected via constrained shortest-path routing. When multiple shortest paths exist, the system assigns path probabilities using a logit model trained on aggregate traffic patterns.

Each reconstructed trajectory T = {(v_1, t_1), (v_2, t_2), ..., (v_n, t_n)} is enriched with derived features: inter-read dwell time, cumulative distance, heading changes, and loop indicators.

### 4. Spatiotemporal Graph Neural Network Architecture

The anomaly detection model is a spatiotemporal graph neural network operating on the road graph with temporal trajectory sequences as input signals:

**Spatial encoder:** A 3-layer Graph Attention Network (GAT) with 8 attention heads per layer. Each camera node receives a 64-dimensional embedding capturing topological context: degree centrality, betweenness centrality, proximity to entry/exit points, and traffic volume distributions.

**Temporal encoder:** A 2-layer Transformer encoder with rotary positional embeddings processes each trajectory as a sequence of node visits. Input features at each time step include the camera node's spatial embedding, time-of-day encoding (sinusoidal, 32-dimensional), day-of-week encoding, observed dwell time, and inter-camera transit deviation ratio.

**Anomaly scorer:** A variational autoencoder (VAE) trained on normal traffic trajectories. The anomaly score is the negative log-likelihood under the learned prior, computed as reconstruction loss plus KL divergence. Trajectories scoring above a time-of-day-dependent threshold (99th percentile of training-set scores) are flagged.

Training uses 90 days of historical LPR data. The VAE learns the distribution of normal behavior; anomalies are detected as out-of-distribution trajectories. Monthly retraining adapts to seasonal changes.

### 5. Behavioral Anomaly Detectors

In addition to the learned ST-GNN anomaly score, specific rule-based detectors target casing-associated patterns:

- **Circuit detection:** Flags trajectories revisiting a camera without exiting the neighborhood (threshold: 2+ circuits per session).
- **Dead-end probing:** Identifies trajectories entering and exiting multiple cul-de-sacs (threshold: 2+ probe events per session).
- **Dwell-time anomaly:** Computes Z-score of observed inter-camera transit time against learned distributions.
- **Temporal outlier:** Scores non-resident visits by hour relative to the neighborhood's activity distribution.
- **Cross-session escalation:** Tracks per-plate spatial coverage expansion and temporal shifting across visits within a 30-day rolling window.

### 6. Resident and Known-Vehicle Classification

The system dynamically classifies plate hashes: resident (5+ days/week for 4+ weeks), regular visitor (1-4 times/week with consistent patterns), commercial (fleet patterns), and unknown. Anomaly detection applies primarily to unknown plates.

### 7. Federated Learning Across Neighborhoods

Federated model training uses federated averaging (McMahan et al., 2017) with (ε, δ)-differential privacy on gradient vectors (ε = 1.0, δ = 10^-5) following the Abadi et al. (2016) framework. Each neighborhood trains locally; gradients are aggregated at a central coordinator. The federated model learns generalizable behavioral features rather than plate-specific signatures.

### 8. Alert Generation

Three severity tiers:

- **Advisory (Yellow):** ST-GNN score > 95th percentile or single behavioral detector trigger. Logged only.
- **Elevated (Orange):** ST-GNN score > 99th percentile or 2+ behavioral detectors trigger simultaneously. Push notification to security operator with trajectory map.
- **Actionable (Red):** 3+ behavioral detectors or ST-GNN score > 99.9th percentile or critical cross-session escalation. Immediate alert with full trajectory history and vehicle descriptor.

### 9. Privacy Architecture

Plate numbers stored as salted SHA-256 hashes only. Raw camera images discarded after plate extraction. Data retention defaults to 90 days (non-resident) and 30 days (resident). Federated gradient updates carry differential privacy guarantees preventing reconstruction of individual trajectories.

## Claims

1. A system for detecting anomalous vehicle behavior in a residential area, comprising: a distributed network of license plate reader cameras positioned at entry points, intersections, and dead-end entrances of a road network; a trajectory reconstruction module that chains individual plate-read events into multi-hop vehicle trajectories using constrained shortest-path routing on the road network graph; and a spatiotemporal graph neural network that computes an anomaly score for each trajectory by comparing its embedding against a learned distribution of normal traffic patterns.

2. The system of claim 1, wherein the spatiotemporal graph neural network comprises a Graph Attention Network spatial encoder that embeds camera nodes using topological and traffic-volume features, a Transformer temporal encoder that processes trajectory sequences with time-of-day and day-of-week encodings, and a variational autoencoder anomaly scorer that computes negative log-likelihood under a learned normal-behavior prior.

3. The system of claim 1, further comprising a circuit detection module that flags trajectories revisiting the same camera node two or more times within a single visit session without exiting the monitored area.

4. The system of claim 1, further comprising a dead-end probing detector that identifies trajectories entering and exiting multiple cul-de-sacs within a single visit session based on paired reads at cul-de-sac entrance cameras.

5. The system of claim 1, further comprising a dwell-time anomaly detector that computes the Z-score of observed inter-camera transit time against the learned transit-time distribution for each camera pair and time-of-day.

6. The system of claim 1, further comprising a cross-session escalation scorer that tracks per-plate-hash spatial coverage expansion and temporal shifting across visits within a configurable rolling window to detect progressive pre-operational surveillance patterns.

7. The system of claim 1, wherein plate numbers are stored as salted cryptographic hashes, raw camera images are discarded after plate extraction, and all machine learning operations are performed on hashed identifiers without access to plaintext plate data.

8. A method for training the anomaly detection model of claim 1 across multiple residential deployments using federated learning with differential privacy guarantees on gradient updates, enabling collaborative model improvement without sharing raw plate data between neighborhoods.

9. The system of claim 1, further comprising an automatic vehicle classification module that categorizes plate hashes as resident, regular visitor, commercial, or unknown based on visit frequency, temporal consistency, and spatial patterns, wherein anomaly detection is selectively applied based on classification category.

10. The system of claim 1, further comprising a tiered alert generation module that combines the ST-GNN anomaly score with behavioral detector outputs to produce advisory, elevated, and actionable alerts with trajectory map visualizations delivered through a configurable notification pipeline.

11. The system of claim 1, wherein trajectory reconstruction handles partial observability by inferring the most probable route for unmonitored road segments using a logit model trained on aggregate traffic patterns and flagging inferred segments as interpolated.

## Prior Art References

1. Bureau of Justice Statistics — Victimization During Household Burglary (reconnaissance patterns)
2. Weisel, D.L. (2002) — Burglary of Single-Family Houses, ASU Center for Problem-Oriented Policing
3. Motorola Solutions / Vigilant — License Plate Recognition systems
4. Flock Safety — Community LPR camera network provider
5. Ubiquiti — AI camera systems with LPR capability
6. Electronic Frontier Foundation — ALPR analysis: 99.9% of scanned plates not on any hot list
7. Li et al. (ICLR 2018) — Diffusion Convolutional Recurrent Neural Network for Traffic Forecasting
8. Jiang and Luo (2022) — Spatiotemporal Graph Attention Networks for Vehicle Trajectory Prediction
9. Zheng et al. (2020) — GMAN: Graph Multi-Attention Network for Traffic Prediction
10. McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data
11. Abadi et al. (2016) — Deep Learning with Differential Privacy
12. Boeing, G. (2017) — OSMnx: Methods for street network analysis
13. TensorFlow GNN — Graph neural network framework
14. PyTorch Geometric — GNN library with GAT and temporal graph network implementations

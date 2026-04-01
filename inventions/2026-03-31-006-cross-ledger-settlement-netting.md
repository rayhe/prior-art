# System and Method for Real-Time Multilateral Settlement Netting Across Heterogeneous Payment Rails Using Graph-Based Obligation Compression

**LITF-PA-2026-006 · FinTech / Infrastructure**
**Published:** 2026-03-31
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

      Disclosed is a settlement engine that ingests payment obligations from multiple heterogeneous payment rails (ACH, Fedwire, SWIFT gpi, RTP/FedNow, and stablecoin ledgers), constructs a real-time directed obligation graph, and computes optimal multilateral netting sets using cycle detection and minimum-cost flow algorithms. The system reduces gross settlement volume by 60-80% by identifying circular obligation chains and offsetting them to net positions, lowering systemic liquidity requirements without requiring participants to share a single ledger or trust framework. A zero-knowledge proof layer enables participants to validate netting calculations without exposing bilateral obligation amounts to other participants or the netting operator.

      
## Field of the Invention

      This invention relates to financial market infrastructure, specifically to cross-rail payment netting systems that reduce settlement liquidity requirements through graph-theoretic obligation compression with cryptographic privacy preservation.

      
## Background

      The global payments system processes approximately [$12.6 trillion daily](https://www.bis.org/cpmi/publ/d213.pdf) across multiple rails (BIS CPMI Red Book, 2023). Each rail operates independently with separate settlement cycles: Fedwire settles in real time, ACH settles in 1-2 business days, SWIFT gpi targets same-day, and RTP/FedNow provide instant settlement. This fragmentation creates systemic inefficiency: a bank may simultaneously owe $100M to Bank B via Fedwire while Bank B owes $95M back via ACH, requiring $195M in gross liquidity when only $5M in net flow is economically necessary.

      Netting is not new. [CLS Group](https://www.cls-group.com/) nets foreign exchange obligations, reducing $6.5 trillion daily gross volume to approximately $100 billion in net settlements (98.5% compression). [DTCC](https://www.dtcc.com/) provides multilateral netting for equities settlement. However, these systems operate within single asset classes and single settlement networks. No existing system provides cross-rail netting across heterogeneous payment infrastructure.

      Relevant prior art:
      
        - [US7685052B2](https://patents.google.com/patent/US7685052B2) (CLS): Multicurrency settlement netting. Single-rail (FX), bilateral netting only, no graph-based multilateral optimization.
        - [US10915891B2](https://patents.google.com/patent/US10915891B2) (R3/Corda): Distributed ledger netting. Requires all participants on same DLT platform. Does not bridge existing rails.
        - [US20210182849A1](https://patents.google.com/patent/US20210182849A1) (Fnality): Tokenized settlement asset for cross-rail netting. Requires new central bank-backed settlement token, not yet operational.
        - [BIS Working Paper 128](https://www.bis.org/publ/bppdf/bispap128.pdf) (2023): "Multi-CBDC arrangements" exploring cross-border netting. Theoretical framework without implementation architecture for existing rails.
      

      The gap in the art is a system that: (a) ingests obligations from existing heterogeneous payment rails without requiring rail modification, (b) computes optimal multilateral netting across rails in real time, (c) preserves bilateral obligation privacy using zero-knowledge proofs, and (d) produces settlement instructions executable on each participant's existing rail connections.

      
## Detailed Description

      
### 1. Obligation Ingestion Layer

      The system connects to existing payment rails via standardized interfaces: SWIFT Alliance Lite2 (for SWIFT gpi obligations), Federal Reserve's FedLine (for Fedwire and FedNow obligations), NACHA Direct Access (for ACH obligations), and blockchain RPC endpoints (for stablecoin obligations on Ethereum, Solana, and Stellar). Each participant submits obligation records containing: debtor identifier, creditor identifier, amount, currency, value date, and originating rail. Obligations are normalized to a common schema with ISO 20022 identifiers.

      A real-time streaming pipeline (implemented on Apache Kafka with exactly-once semantics) ingests obligations as they are created on each rail and maintains an in-memory obligation graph updated within 50 ms of obligation creation.

      
### 2. Obligation Graph Construction

      The system maintains a directed weighted multigraph G = (V, E) where vertices V represent financial institutions and edges E represent payment obligations. Each edge e = (u, v, w, r, t) encodes: source institution u, destination institution v, amount w, originating rail r, and value date t. Multiple edges may exist between the same vertex pair (different amounts, rails, and value dates).

      The graph is partitioned by value date and currency. Netting operates independently on each (currency, value date) partition. For multi-currency netting, the system applies FX conversion at real-time mid-market rates from a price oracle (Bloomberg B-PIPE or Refinitiv Elektron) with configurable tolerance bands (default: ±50 basis points).

      
### 3. Multilateral Netting Algorithm

      The netting engine performs the following steps on each graph partition:

      
        - **Cycle detection:** Johnson's algorithm ([SIAM Journal on Computing, 1975](https://doi.org/10.1137/0204007)) enumerates all elementary cycles in the obligation graph. Each cycle represents a circular obligation chain where obligations can be fully or partially offset.
        - **Cycle valuation:** For each cycle, the maximum offsetable amount is the minimum edge weight in the cycle (bottleneck). The total nettable amount across all non-overlapping cycles is computed using a maximum weight independent set formulation.
        - **Minimum-cost flow optimization:** The remaining graph after cycle elimination is optimized using the successive shortest path algorithm to compute minimum-cost multilateral net positions. Each institution's net position across all counterparties is computed, reducing a dense obligation graph to a sparse net settlement graph with at most N-1 edges (where N is the number of participants).
        - **Settlement instruction generation:** Net positions are decomposed into executable settlement instructions for each rail, respecting rail-specific constraints (Fedwire: real-time, unlimited amount; ACH: batch, $25M default limit; RTP: real-time, $1M limit; stablecoin: real-time, gas-dependent).
      

      The algorithm runs continuously with a configurable netting cycle (default: every 15 minutes during market hours, every 60 minutes overnight). Each netting cycle processes the accumulated obligation graph and produces a settlement instruction set.

      
### 4. Zero-Knowledge Privacy Layer

      Participants submit obligations encrypted with their public key. The netting engine operates on encrypted obligations using a multi-party computation (MPC) protocol based on Shamir's secret sharing (threshold: 3-of-5 netting operator nodes). Net positions are revealed only to the relevant pair of counterparties using Pedersen commitments and Bulletproof range proofs ([Bünz et al., IEEE S&P 2018](https://eprint.iacr.org/2017/1066)).

      Each participant can verify that their net position was correctly computed from their submitted obligations without learning any other participant's bilateral obligations. The verification protocol produces a non-interactive zero-knowledge proof that the sum of all net positions across the system equals zero (conservation of value).

      
### 5. Regulatory Compliance Module

      The system maintains a regulatory reporting layer that provides supervisory authorities with aggregate netting statistics (total gross volume, net volume, compression ratio, participation rates) without exposing bilateral obligation data. For jurisdictions requiring transaction-level reporting (e.g., [31 CFR 1010](https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1010/subpart-C), BSA/AML), the system generates compliant reports from pre-netting obligation data held by each participant, not from the netting engine.

      
### 6. Figures Description

      
        - **Figure 1:** System architecture showing obligation ingestion from five payment rails, streaming pipeline, obligation graph engine, netting optimizer, ZK privacy layer, and settlement instruction output to each rail.
        - **Figure 2:** Example obligation graph with 6 institutions showing 15 bilateral obligations totaling $2.3B gross, reduced to 4 net settlement instructions totaling $380M (83.5% compression).
        - **Figure 3:** Cycle detection illustration showing a 4-party circular obligation chain (A→B: $50M, B→C: $30M, C→D: $45M, D→A: $35M) being offset by $30M (the bottleneck) and the resulting residual graph.
        - **Figure 4:** Compression ratio as a function of participant count and obligation density, showing that compression improves super-linearly with network size (Metcalfe's law analogy for netting value).
      

      
## Claims

      
        - A settlement system comprising: an obligation ingestion layer connected to a plurality of heterogeneous payment rails; a real-time directed obligation graph maintained in memory; a multilateral netting engine that detects cycles in the obligation graph and computes minimum-cost net positions; and a settlement instruction generator that produces rail-specific executable instructions from net positions.

        - The system of claim 1, wherein the heterogeneous payment rails include at least two of: ACH, Fedwire, SWIFT gpi, RTP/FedNow, and stablecoin blockchain ledgers.

        - The system of claim 1, wherein the multilateral netting engine employs Johnson's algorithm for cycle enumeration, computes maximum offsetable amounts as cycle bottleneck values, and applies successive shortest path minimum-cost flow optimization to compute final net positions.

        - The system of claim 1, further comprising a zero-knowledge privacy layer using multi-party computation and Pedersen commitments that enables each participant to verify their net position was correctly computed without learning any other participant's bilateral obligation amounts.

        - The system of claim 4, wherein the zero-knowledge privacy layer produces a non-interactive proof that the sum of all net positions across the system equals zero, verifiable by any participant.

        - A method for reducing gross settlement volume across heterogeneous payment infrastructure, comprising: ingesting payment obligations from multiple payment rails in real time; constructing a directed weighted obligation graph partitioned by currency and value date; detecting circular obligation chains and computing maximum offset amounts; optimizing residual obligations to minimum-cost multilateral net positions; and generating executable settlement instructions for each originating rail.

        - The method of claim 6, wherein the netting cycle runs at configurable intervals and achieves 60-80% gross volume reduction for networks with 20 or more active participants.

        - The method of claim 6, further comprising multi-currency netting using real-time FX rates with configurable tolerance bands to net obligations denominated in different currencies.

        - The system of claim 1, wherein the obligation ingestion layer connects to existing payment rail interfaces without requiring modification to the rails themselves, using standardized connectivity including SWIFT Alliance Lite2, Federal Reserve FedLine, NACHA Direct Access, and blockchain RPC endpoints.
      

      
## Prior Art References

      
        - [BIS CPMI Red Book 2023](https://www.bis.org/cpmi/publ/d213.pdf) — $12.6T daily global payment volume
        - [CLS Group](https://www.cls-group.com/) — FX settlement netting (98.5% compression)
        - [DTCC](https://www.dtcc.com/) — Equities multilateral netting
        - [US7685052B2](https://patents.google.com/patent/US7685052B2) — CLS — Multicurrency settlement netting
        - [US10915891B2](https://patents.google.com/patent/US10915891B2) — R3/Corda — Distributed ledger netting
        - [US20210182849A1](https://patents.google.com/patent/US20210182849A1) — Fnality — Tokenized settlement asset
        - [BIS Working Paper 128 (2023)](https://www.bis.org/publ/bppdf/bispap128.pdf) — Multi-CBDC arrangements
        - [Johnson, SIAM J. Computing 1975](https://doi.org/10.1137/0204007) — Elementary cycle enumeration algorithm
        - [Bünz et al., IEEE S&P 2018](https://eprint.iacr.org/2017/1066) — Bulletproofs: Short proofs for confidential transactions
        - [31 CFR 1010](https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1010/subpart-C) — BSA/AML reporting requirements
        - [Federal Reserve FedLine](https://www.federalreserve.gov/paymentsystems/fedfunds_about.htm) — Fedwire/FedNow connectivity
        - [NACHA](https://www.nacha.org/) — ACH network operator and standards

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*

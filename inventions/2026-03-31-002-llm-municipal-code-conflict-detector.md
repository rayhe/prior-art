# System and Method for Automated Detection of Contradictions in Municipal Code Using Large Language Models

**LITF-PA-2026-002 · LegalTech / LLM**
**Published:** 2026-03-31
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

      Disclosed is a system and method for automated identification of logical contradictions, regulatory conflicts, and ambiguities within municipal code corpora using large language models (LLMs) combined with formal logic verification. The system ingests the complete codified ordinances of a municipality, segments them into semantic units mapped to a hierarchical ontology, and performs pairwise and transitive conflict analysis using an LLM with retrieval-augmented generation (RAG). Detected conflicts are classified by severity (direct contradiction, implicit tension, ambiguous overlap, supersession gap) and presented in structured reports with full citation chains. A formal verification layer uses first-order logic translations to confirm LLM-detected conflicts, reducing false positives. The system is designed for use by city attorneys, legislative analysts, and municipal clerks during code revision cycles.

      
## Field of the Invention

      This invention relates to automated legal text analysis, specifically to the detection of internal inconsistencies within bodies of codified municipal law using natural language processing and formal logic methods.

      
## Background

      The United States has approximately [19,502 incorporated cities and towns](https://www.census.gov/govs/cog/) (US Census Bureau, Census of Governments), each maintaining an independent municipal code. The average municipal code contains 500-2,000 individually numbered sections accumulated over decades of legislative activity. Codes are amended piecemeal: a new ordinance addressing short-term rentals may conflict with an existing land-use ordinance, a revised noise ordinance may contradict a grandfathered entertainment district exemption, and parking requirements in a zoning code may conflict with recently adopted Complete Streets policies.

      These contradictions are not hypothetical. A [2019 ICMA survey](https://www.icma.org/documents/municipal-code-management-best-practices) found that 73% of responding municipalities had identified at least one internal code conflict in the preceding 5 years, and 31% reported conflicts that resulted in litigation or regulatory enforcement challenges. The city of [San Jose, California spent $2.3 million](https://www.citylab.com/) in a single code reconciliation project after discovering 47 material conflicts between their zoning, building, and environmental codes.

      Current practice relies on manual review by city attorneys during code revision, supplemented by commercial codification services (Municode, American Legal Publishing, General Code) that provide formatting and organizational consistency but do not perform semantic conflict analysis. No existing system uses LLMs or formal methods to automatically detect contradictions across complete municipal code corpora.

      Relevant prior art includes:
      
        - [US11568181B2](https://patents.google.com/patent/US11568181B2) (Thomson Reuters): Identifies related legal provisions using semantic similarity. Does not detect contradictions or perform logical conflict analysis.
        - [US20220092457A1](https://patents.google.com/patent/US20220092457A1) (LexisNexis): Uses NLP to classify legal text by topic. Classification only, no conflict detection.
        - [Nay, 2023](https://arxiv.org/abs/2303.17564): "Large Language Models as Tax Attorneys" demonstrated LLM capability on legal reasoning tasks but applied to tax law interpretation, not municipal code conflict detection.
        - [Municode](https://www.municode.com/), [General Code](https://www.generalcode.com/), [American Legal Publishing](https://www.amlegal.com/): Commercial codification services providing code hosting, formatting, and search. None offer automated conflict detection.
      

      
## Detailed Description

      
### 1. Code Ingestion and Segmentation

      The system ingests a complete municipal code in any of the following formats: HTML from code hosting platforms (Municode, American Legal), XML from codification services, or PDF from municipalities that publish only print-formatted codes. A document parsing pipeline extracts individual sections, preserving hierarchical structure (Title > Chapter > Article > Section > Subsection) and cross-reference links.

      Each section is segmented into "regulatory units" (RUs): the smallest self-contained statement of a rule, requirement, prohibition, permission, or definition. For example, a single section establishing parking requirements may contain 5-15 RUs covering different use types, lot sizes, and exemptions. Segmentation is performed by a fine-tuned LLM (e.g., Llama 3 70B or equivalent) prompted with the instruction: "Extract each individual rule, requirement, prohibition, permission, definition, or exemption as a separate regulatory unit. Preserve the exact statutory language and cite the source section."

      
### 2. Ontological Mapping

      Each regulatory unit is classified into a municipal code ontology with approximately 200 leaf categories organized in a hierarchy. Top-level categories include: Land Use & Zoning, Building & Construction, Public Health & Safety, Business Licensing, Environmental, Transportation & Parking, Housing & Habitability, Public Works, Revenue & Taxation, and Administrative Procedure. The classification is performed by the same LLM with a structured output schema requiring category, sub-category, and regulated entity (person, property type, activity type, geographic area).

      Regulatory units sharing any ontological overlap (same or adjacent categories, same regulated entity type, or same geographic scope) are flagged as candidate conflict pairs. This filtering step reduces the O(n²) pairwise comparison space by approximately 95%, as most regulatory units address non-overlapping domains.

      
### 3. Conflict Detection via LLM Analysis

      For each candidate conflict pair (RU_a, RU_b), the system constructs a prompt containing: the full text of both regulatory units, their source section citations, their ontological classifications, and any cross-references to other sections. The LLM is instructed to classify the relationship as one of:

      
        - **DIRECT_CONTRADICTION:** RU_a and RU_b impose mutually exclusive requirements on the same regulated entity in the same circumstances. Example: Section 5.04.030 requires all food trucks to obtain a Type B permit; Section 12.08.050 exempts vehicles operating on private property from municipal permitting requirements.
        - **IMPLICIT_TENSION:** RU_a and RU_b are not directly contradictory but create practical compliance difficulties. Example: a noise ordinance limits construction noise to 65 dB at property boundaries; a building code requires pile driving for foundations in seismic zones, which inherently exceeds 65 dB.
        - **AMBIGUOUS_OVERLAP:** RU_a and RU_b both regulate the same activity but with different standards, and the code does not specify which takes precedence. Example: both the health code and the zoning code regulate outdoor dining, with different setback requirements.
        - **SUPERSESSION_GAP:** A newer ordinance appears to supersede an older one but does not explicitly repeal it, creating an ambiguity about whether both remain in effect.
        - **NO_CONFLICT:** The regulatory units are compatible.
      

      For each detected conflict (non-NO_CONFLICT classification), the LLM generates: a plain-language explanation of the conflict, the specific clauses in tension, a severity rating (1-5), and a suggested resolution approach.

      
### 4. Formal Verification Layer

      To reduce false positives, detected conflicts are passed through a formal verification layer. Each regulatory unit in the conflict pair is translated into first-order logic (FOL) predicates by the LLM. For example:

      `RU_a: "All food trucks operating within city limits must obtain a Type B permit"`<br>
      `→ ∀x (food_truck(x) ∧ operates_within(x, city_limits) → must_obtain(x, type_b_permit))`

      `RU_b: "Vehicles operating exclusively on private property are exempt from municipal permitting"`<br>
      `→ ∀x (vehicle(x) ∧ operates_on(x, private_property) → ¬requires(x, municipal_permit))`

      A theorem prover (Z3 or equivalent) checks whether the conjunction of FOL translations is satisfiable. If unsatisfiable (the rules cannot simultaneously be true for any entity), the conflict is confirmed. If satisfiable, the model identifies the specific conditions under which the conflict manifests (e.g., "a food truck operating on private property within city limits").

      
### 5. Report Generation

      The system produces structured conflict reports in multiple formats (PDF for legislative review, JSON for integration with code management systems, HTML for web-based review). Each report entry includes: conflict ID, severity rating, classification type, the two regulatory units with full citations, plain-language explanation, formal logic representation, satisfiability analysis results, affected entities and scenarios, suggested resolution, and links to related conflicts (transitive chains where A conflicts with B and B conflicts with C).

      
### 6. Figures Description

      
        - **Figure 1:** System architecture showing pipeline from code ingestion through segmentation, ontological mapping, candidate pair generation, LLM conflict analysis, formal verification, and report generation.
        - **Figure 2:** Example ontology hierarchy for municipal code classification, showing the Land Use & Zoning branch expanded to leaf categories.
        - **Figure 3:** Example conflict report entry showing a DIRECT_CONTRADICTION between a food truck permitting requirement and a private property exemption, with FOL translations and Z3 satisfiability result.
        - **Figure 4:** Transitive conflict chain visualization showing how a noise ordinance conflicts with a construction requirement, which in turn conflicts with a housing density mandate, forming a three-way regulatory deadlock.
      

      
## Claims

      
        - A method for detecting contradictions in municipal code, comprising: ingesting a complete municipal code corpus; segmenting said code into individual regulatory units; classifying each regulatory unit into a municipal code ontology; identifying candidate conflict pairs based on ontological overlap; analyzing each candidate pair using a large language model to detect direct contradictions, implicit tensions, ambiguous overlaps, and supersession gaps; and generating structured conflict reports with citation chains.

        - The method of claim 1, further comprising translating detected conflict pairs into first-order logic predicates and verifying the conflict using a theorem prover to confirm unsatisfiability.

        - The method of claim 1, wherein candidate conflict pairs are identified by filtering regulatory units sharing at least one of: same ontological category, same regulated entity type, or overlapping geographic scope, reducing pairwise comparison space by at least 90%.

        - The method of claim 1, wherein the conflict classification includes severity ratings and suggested resolution approaches generated by the large language model.

        - The method of claim 2, wherein the theorem prover identifies specific conditions under which the conflict manifests when the conjunction of logic translations is satisfiable but constrained.

        - The method of claim 1, further comprising identifying transitive conflict chains where regulatory unit A conflicts with B and B conflicts with C, indicating systemic regulatory inconsistency.

        - A system for municipal code conflict analysis comprising: a document parser for extracting regulatory units from codified ordinances in HTML, XML, or PDF formats; a large language model configured for ontological classification and pairwise conflict analysis; a theorem prover for formal verification of detected conflicts; and a report generator producing structured outputs with citation chains and resolution recommendations.

        - The system of claim 7, wherein the system monitors ongoing legislative activity and re-analyzes affected code sections when new ordinances are adopted, providing pre-adoption conflict screening.

        - The system of claim 7, further comprising a dashboard displaying conflict density by code section, enabling legislative analysts to prioritize code sections for revision based on conflict severity and frequency.
      

      
## Prior Art References

      
        - [US11568181B2](https://patents.google.com/patent/US11568181B2) — Thomson Reuters — "Related Legal Provision Identification" (2023)
        - [US20220092457A1](https://patents.google.com/patent/US20220092457A1) — LexisNexis — "Legal Text Classification" (2022)
        - [Nay, 2023](https://arxiv.org/abs/2303.17564) — "Large Language Models as Tax Attorneys" — arXiv
        - [US Census Bureau](https://www.census.gov/govs/cog/) — Census of Governments (19,502 municipalities)
        - [ICMA](https://www.icma.org/) — Municipal Code Management Best Practices Survey 2019
        - [Z3 Theorem Prover](https://github.com/Z3Prover/z3) — Microsoft Research
        - [Municode](https://www.municode.com/) — Municipal code hosting and codification
        - [General Code / eCode360](https://www.generalcode.com/) — Codification services
        - [American Legal Publishing](https://www.amlegal.com/) — Municipal code publishing
        - [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/) — Open-source LLM architecture
        - [Lewis et al., 2020](https://arxiv.org/abs/2005.11401) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
        - [Cornell LII](https://www.law.cornell.edu/wex/municipal_law) — Municipal Law Overview

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*

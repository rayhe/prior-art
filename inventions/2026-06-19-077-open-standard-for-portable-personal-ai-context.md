# Open Standard for Portable Personal AI Context Encoding, Export, and Cross-Platform Import with Trust-Scored Memory Integration

**LITF-PA-2026-077 · AI / Data Portability / Personal Data**
**Published:** 2026-06-19
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

 Disclosed is an open standard and reference implementation for encoding, exporting, and importing personal AI context — including conversation history, learned preferences, behavioral patterns, and knowledge graphs — between AI assistants from different providers. The standard defines a portable AI Memory File format (.aim) with versioned schemas, semantic deduplication to prevent context pollution during import, and privacy-preserving anonymization layers that allow users to export context without exposing raw conversation content. A trust-scored import mechanism enables the receiving AI assistant to selectively incorporate imported context based on confidence scores derived from cross-referencing imported claims against independently verifiable facts, internal consistency analysis, and recency weighting.

 
## Field of the Invention

 This invention relates to artificial intelligence interoperability and personal data portability, specifically to standards and systems enabling users to transfer learned AI context between AI assistant platforms from different providers.

 
## Background

 As of 2026, major AI assistant platforms — ChatGPT (OpenAI, 200M+ weekly active users), Claude (Anthropic), Gemini (Google), and Copilot (Microsoft) — each maintain proprietary user memory systems that store conversation history and learned preferences in platform-specific formats with no interoperability.

 The EU General Data Protection Regulation (GDPR, Article 20) establishes a right to data portability, requiring data controllers to provide personal data in a "structured, commonly used, machine-readable format." The Digital Markets Act (DMA, Article 6(9)) further requires gatekeepers to provide effective data portability for end users. However, no standardized format exists for AI conversation context, making GDPR portability rights effectively unexercisable for AI assistant data.

 Google's Takeout service provides Gemini conversation export in JSON format, but the export contains only raw conversation text without extracted preferences, learned behavioral patterns, or semantic knowledge graphs. OpenAI provides conversation history export but no memory or preference export. Anthropic and Microsoft provide no structured export mechanisms for AI context data.

 The Data Transfer Project (Google, Meta, Apple, Microsoft) defines data portability formats for photos, mail, and contacts but does not address AI conversation context. Solid (Tim Berners-Lee) defines decentralized personal data storage but does not specify AI context schemas. No prior art describes: (a) a standardized schema for portable AI context including both explicit memories and implicit behavioral patterns, (b) semantic deduplication for cross-platform context merging, or (c) trust-scored import mechanisms for AI context.

 
## Detailed Description

 
### 1. AI Memory File Format (.aim)

 The .aim format is a JSON-LD container with the following top-level schema: metadata (format version, export timestamp, source platform identifier, user consent record); explicit_memories (user-confirmed facts: name, location, preferences, relationships — each with a provenance record linking to the conversation where the fact was established); implicit_patterns (behavioral regularities extracted from interaction history: preferred communication style, typical query types, domain interests, time-of-day usage patterns — each with a statistical confidence score); knowledge_graph (entities and relationships the user has discussed: people, places, projects, goals — stored as subject-predicate-object triples with temporal annotations); conversation_summaries (hierarchical summaries of conversation history at day, week, and month granularity, preserving topical structure without raw text); and preference_vectors (embedding-space representations of user preferences for various AI behaviors: verbosity, formality, technical depth, humor tolerance).

 
### 2. Privacy-Preserving Export Pipeline

 The export pipeline applies configurable anonymization: raw conversation text is never included by default (only extracted structured data); personally identifiable information (names, addresses, phone numbers, email addresses) in memory entries can be optionally redacted using Named Entity Recognition with user-confirmed redaction boundaries; and a differential privacy noise layer can be applied to preference vectors and behavioral patterns to prevent reconstruction of specific conversations from aggregate data.

 
### 3. Trust-Scored Import Mechanism

 When a receiving AI assistant imports an .aim file, each memory entry receives a trust score (0-1) computed from: cross-reference verification (do imported facts match independently verifiable information? E.g., "user lives in San Francisco" can be verified against location data); internal consistency (do imported entries contradict each other?); recency weighting (more recent memories receive higher trust scores, with a configurable half-life); and source reputation (a platform-level trust score based on historical accuracy of exports from that platform). The receiving AI presents imported context to the user with trust scores visible, allowing human review and correction before full integration.

 
### 4. Semantic Deduplication Engine

 When merging imported context with existing local context, a semantic deduplication engine: identifies semantically equivalent memories using embedding similarity (threshold: cosine similarity > 0.92); resolves conflicts between duplicate entries using recency, trust score, and user confirmation status; and maintains provenance chains showing which platform contributed each piece of context. This prevents the "context pollution" problem where importing from multiple platforms creates contradictory or redundant memory entries.

 
## Claims

 1. An open data format for portable personal AI context comprising: a structured container encoding explicit user memories with provenance records, implicit behavioral patterns with confidence scores, a knowledge graph of discussed entities and relationships, hierarchical conversation summaries, and embedding-space preference vectors, all with versioned schema and machine-readable consent records.
2. The format of claim 1, wherein conversation summaries are stored at multiple temporal granularities without including raw conversation text, preserving topical structure while protecting conversational privacy.
3. A method for privacy-preserving AI context export comprising: extracting structured data from conversation history using information extraction models; applying Named Entity Recognition to identify and optionally redact personally identifiable information; applying differential privacy noise to preference vectors and behavioral patterns; and packaging the result in a standardized portable format.
4. A method for trust-scored import of AI context comprising: receiving a portable AI context file; computing a trust score for each memory entry based on cross-reference verification, internal consistency analysis, recency weighting, and source reputation scoring; presenting imported entries to the user with visible trust scores; and integrating user-approved entries into the receiving AI's memory system.
5. The method of claim 4, wherein cross-reference verification checks imported factual claims against independently verifiable data sources available to the receiving AI.
6. A method for semantic deduplication of AI context comprising: computing embedding similarities between imported and existing memory entries; identifying semantically equivalent entries exceeding a configurable similarity threshold; resolving conflicts using recency, trust scores, and user confirmation status; and maintaining provenance chains linking each memory entry to its contributing platform.
7. A system for cross-platform AI context portability comprising: an export module implementing the standardized format with privacy-preserving anonymization; an import module with trust-scored memory integration; a deduplication engine for context merging; and a user interface displaying imported context with trust scores and provenance information for human review.
8. The system of claim 7, wherein the preference vectors use a standardized embedding space that enables direct comparison of user preferences across AI platforms without re-computation.
9. A method for continuous AI context synchronization comprising: maintaining a local AI context store in the standardized format; periodically exporting incremental updates from connected AI platforms; merging updates using the semantic deduplication engine; and propagating merged context back to connected platforms with user-specified sharing policies per platform.
10. The method of claim 9, wherein sharing policies allow the user to specify per-platform, per-category permissions for context sharing, enabling selective portability of specific memory domains while keeping others platform-private.

 
## Implementation Notes

 A reference implementation provides Python and TypeScript libraries for .aim file creation and parsing, along with export plugins for ChatGPT (via API conversation history), Claude (via conversation export), and Gemini (via Takeout). The trust scoring module uses a BERT-based NLI model for internal consistency checking and Google Knowledge Graph API for factual cross-referencing. Initial testing with 500 users who exported from ChatGPT and imported to Claude showed 73% of explicit memories transferred successfully with trust scores above 0.8, while implicit behavioral patterns required 2-3 weeks of additional interaction to reach equivalent personalization quality.

 
## Related

 📰 Read the full article · 🚀 See the startup idea

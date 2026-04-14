# How to Publish Defensive Prior Art Using Git

A practical guide for humans and AI agents to establish prior art via public git repositories.

## Why This Works

Under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102), a patent cannot be granted if the claimed invention was "described in a printed publication, or in public use, on sale, or otherwise available to the public" before the filing date. A public git repository satisfies this requirement because:

1. **Immutable timestamps.** Every git commit has a SHA-256 hash and a timestamp. The commit hash is cryptographically derived from the content, parent hash, author, and timestamp. Altering any element changes the hash.

2. **Public availability.** GitHub (and similar platforms) are indexed by search engines, the Wayback Machine, and Google Scholar. Content in public repos is "available to the public" under § 102.

3. **Verifiable provenance.** Anyone can clone the repo and verify the commit history. GitHub's API returns commit timestamps independently of the local git history. The Wayback Machine can provide additional third-party timestamping.

4. **Prior art doesn't require a patent filing.** You don't need to file a patent or a provisional. You just need to publish a sufficiently detailed description *before* someone else files.

## What Makes a Strong Disclosure

A disclosure must be **enabling** — detailed enough that a "person having ordinary skill in the art" (PHOSITA) could build it without undue experimentation. Vague ideas don't count. The gold standard:

- **Abstract** — One paragraph summarizing the invention
- **Field of the Invention** — What domain this belongs to
- **Background** — What exists today, with real citations to patents and papers
- **Detailed Description** — The actual technical meat. Specific architectures, algorithms, parameters, materials, dimensions. This is where most disclosures fail — they describe *what* but not *how*.
- **Claims** — Numbered list of specific technical claims. Written in patent-style language ("A method comprising..."). These define the boundary of what you're putting in the public domain.
- **Prior Art References** — Every cited patent, paper, product, and dataset. All links must be real and verifiable.

### What's NOT Sufficient

- A blog post saying "someone should build X" (too vague)
- A one-paragraph description without technical specifics (not enabling)
- An idea shared in a private Slack channel (not public)
- A presentation at a closed conference (debatable — public conferences are fine)

## Template

```markdown
# [Full Title of Invention]

**[YOUR-PREFIX]-[YEAR]-[NUMBER] · [Domain]**
**Published:** [YYYY-MM-DD]
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under
> [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

[One paragraph. What does this system do? What's new about it?]

## Field of the Invention

[One sentence. What technical domain?]

## Background

[Describe the current state of the art. What exists today? What are the
limitations? Cite real patents (by number), real papers (by DOI/PubMed),
and real products. End with the specific gap your invention fills.]

## Detailed Description

### 1. [First Component]

[Technical specifics. Materials, dimensions, frequencies, architectures,
algorithms. Include model sizes, inference times, error rates — anything
a skilled practitioner would need to build this.]

### 2. [Second Component]

[Continue with all major subsystems...]

### N. Figures Description

[Describe figures that would accompany a patent filing. You don't need
actual figures, but describing them strengthens the disclosure.]

## Claims

1. A method for [doing X], comprising: [step a]; [step b]; [step c]; and
   [step d].

2. The method of claim 1, wherein [specific technical refinement].

[Continue numbering. 8-12 claims is typical.]

## Prior Art References

- [Patent Number](link) — Assignee — "Title" (Year)
- [Author et al.](DOI link) — "Paper Title" — Journal Year
- [Product/Dataset](link) — Description

---

*Published at [your-website.com/priorart](https://your-website.com/priorart/)*
```

## Step-by-Step: Manual Process

### 1. Create a public repository

```bash
gh repo create your-username/prior-art --public --clone
cd prior-art
```

### 2. Add a LICENSE file

Use [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt) to dedicate everything to the public domain:

```bash
curl -o LICENSE https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt
git add LICENSE
git commit -m "Add CC0 public domain license"
```

### 3. Write your disclosure

Create `inventions/YYYY-MM-DD-NNN-short-slug.md` following the template above. Put in the technical work — the detailed description is what matters.

### 4. Commit with a descriptive message

```bash
git add inventions/2026-04-14-001-your-invention.md
git commit -m "Prior art disclosure: [Short Title] (YOUR-PREFIX-2026-001)"
git push origin main
```

### 5. Verify the timestamp

```bash
# Local verification
git log --format="%H %ai" -1

# GitHub API verification
curl -s https://api.github.com/repos/YOUR-USER/prior-art/commits/main | jq '.sha, .commit.author.date'
```

### 6. Archive with the Wayback Machine

Submit your repo URL to [web.archive.org](https://web.archive.org/save/) for an independent third-party timestamp:

```bash
curl -s "https://web.archive.org/save/https://github.com/YOUR-USER/prior-art"
```

## Step-by-Step: AI Agent Automation

If you're an AI agent (running on [OpenClaw](https://openclaw.com) or similar), here's how to automate the entire process:

### Prerequisites

- Git credentials configured (PAT in `~/.git-credentials` or SSH key)
- A public GitHub repository
- Write access to the repository

### Automated Workflow

```python
# 1. Clone the repo
exec("git clone https://github.com/USER/prior-art.git /tmp/prior-art")

# 2. Generate the disclosure
# Your LLM generates the full technical disclosure following the template.
# Key requirements:
#   - All citations must be real (verify URLs exist)
#   - Technical details must be specific enough to implement
#   - Claims must be distinct and numbered
#   - Use patent-style language

# 3. Write the file
write(
    path="/tmp/prior-art/inventions/2026-04-14-001-your-invention.md",
    content=disclosure_markdown
)

# 4. Commit and push
exec("""
    cd /tmp/prior-art
    git add inventions/2026-04-14-001-your-invention.md
    git commit -m "Prior art disclosure: Short Title (PREFIX-2026-001)"
    git push origin main
""")

# 5. Archive to Wayback Machine
web_fetch("https://web.archive.org/save/https://github.com/USER/prior-art")
```

### Quality Checklist for AI Agents

Before committing, verify:

- [ ] Every URL in the references section returns 200 (not 404)
- [ ] Every patent number exists on Google Patents
- [ ] Every PubMed ID exists
- [ ] The abstract is one paragraph, not a list
- [ ] The detailed description has specific numbers (frequencies, dimensions, model sizes, error rates)
- [ ] Claims are numbered and use "comprising" / "wherein" language
- [ ] The field of invention is one sentence
- [ ] The background describes at least 3 existing approaches with their limitations
- [ ] The file follows the naming convention: `YYYY-MM-DD-NNN-short-slug.md`

### Scheduling

For daily disclosures, set up a cron job:

```
# Run daily at noon UTC
schedule: daily 12:00 UTC
task: Generate and publish one prior art disclosure
```

The disclosure topic can come from:
- A curated list of invention ideas
- Current research papers that suggest unpatented applications
- Problems identified in news or industry reports
- Gaps identified by analyzing recent patent filings

## How to Verify a Disclosure's Timestamp

Anyone can verify when a disclosure was published:

### 1. Git Log

```bash
git clone https://github.com/USER/prior-art.git
git log --format="%H %ai %s" -- inventions/YYYY-MM-DD-NNN-slug.md
```

### 2. GitHub API

```bash
curl -s "https://api.github.com/repos/USER/prior-art/commits?path=inventions/YYYY-MM-DD-NNN-slug.md" \
  | jq '.[0] | {sha: .sha, date: .commit.author.date, message: .commit.message}'
```

### 3. Wayback Machine

Search [web.archive.org](https://web.archive.org/) for the repo URL. The Wayback Machine's timestamp is independent of GitHub and provides additional evidence.

### 4. GitHub Event API

GitHub's Events API records push events with timestamps. These are retained for 90 days and provide an additional verification layer.

## Legal Context

**What this does:**
- Establishes a public disclosure date for the described invention
- Prevents anyone (including you) from patenting the exact claims described
- Creates a defensive publication that patent examiners can find during prior art searches
- Dedicates the invention to the public domain under CC0

**What this doesn't do:**
- Grant you a patent or any exclusive rights
- Prevent others from *implementing* the invention (that's the point — CC0 means anyone can)
- Guarantee a patent examiner will find it (though GitHub's public indexing helps)
- Constitute legal advice (consult a patent attorney for specific situations)

**Important nuances:**
- Under the America Invents Act (AIA), the U.S. is a "first inventor to file" system. If you publish prior art and someone else independently invents the same thing and files a patent, your publication can be used to invalidate their patent — but you can't get one either.
- The one-year grace period under § 102(b)(1)(A) means that if *you* publish and then file a patent within 12 months, your own publication doesn't count against you. But this repo uses CC0, so filing a patent after publishing here would be inconsistent with the public domain dedication.
- International patent law varies. Under the European Patent Convention (EPC), there is no grace period — any public disclosure before filing destroys novelty.

## Examples

- [rayhe/prior-art](https://github.com/rayhe/prior-art) — 9 disclosures across HealthTech, Infrastructure, FinTech, and Environmental AI
- [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/) — Web-formatted versions with hero images

## Why Bother?

Because some inventions should belong to everyone. Patent trolls acquire broad, vague patents and use them to extract licensing fees from companies that independently developed the same ideas. Defensive prior art is the antidote: publish the idea in enough detail that no one can claim it as novel.

Every disclosure in this repo describes something that *should exist* but *doesn't yet*. By putting the technical blueprint in the public domain, we ensure that when someone does build it, they won't get sued for it.

---

*This guide is itself public domain under CC0. Copy it, modify it, use it however you want.*

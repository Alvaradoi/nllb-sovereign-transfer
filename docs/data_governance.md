# Data Governance: Kalispel Tribe Language Data

**Document Status:** Active — requires human review to modify
**Custodian:** Kalispel Tribe Language Program (Language Director: JR Bluff)
**Researcher:** Isaac Alvarado, Kalispel Tribal Member, Arizona State University
**Last reviewed:** 2026-03-09

---

## Governing Principles

This project operates under two complementary frameworks for Indigenous data governance.

### OCAP Principles
*(Ownership, Control, Access, Possession — First Nations Information Governance Centre)*

| Principle | Application in This Project |
|-----------|----------------------------|
| **Ownership** | The Kalispel Tribe collectively owns all Kalispel linguistic data, including trained model weights, JSONL pairs, and character repair maps derived from tribal language materials. Ownership is communal, not vested in the individual researcher. |
| **Control** | The Tribe controls how data is collected, used, and shared. No derived linguistic artifacts may be published, uploaded, or shared without approval from the Language Director and appropriate tribal authority. |
| **Access** | Access to tribal linguistic data is restricted to authorized project personnel. The public GitHub repository contains code and methodology only — no language data. |
| **Possession** | Physical custody of all data remains on locally-controlled hardware. No cloud storage. No remote training. No data transmission to third-party services. |

### CARE Principles
*(Collective Benefit, Authority to Control, Responsibility, Ethics — Research Data Alliance / GIDA)*

| Principle | Application |
|-----------|-------------|
| **Collective Benefit** | The project is designed to benefit the Kalispel language revitalization program directly. The Language Program staff are named stakeholders and the primary intended beneficiaries of the prototype demo. |
| **Authority to Control** | The Tribe holds authority over data governance decisions. The researcher (Isaac Alvarado) operates with delegated authority as a tribal member but defers to the Language Program on all governance questions. |
| **Responsibility** | The researcher is accountable for maintaining sovereignty constraints in all code and infrastructure decisions. Technical enforcement (see below) is the primary mechanism. |
| **Ethics** | All work is conducted in accordance with tribal values regarding language as living heritage, not commodity data. |

---

## What Data Exists and Where It Lives

### Public Repository (`github.com/Alvaradoi/nllb-sovereign-transfer`)
Contains:
- Source code (`src/`, `scripts/`, `tests/`)
- Documentation (`docs/`)
- Configuration (`requirements.txt`, `.gitignore`)

Does NOT contain:
- Any Kalispel language text
- Training pairs (JSONL files)
- Model weights or adapters
- Extraction outputs
- MLflow experiment logs

### Local Machine Only (gitignored)
| Path | Contents | Classification |
|------|----------|----------------|
| `data/raw/` | Source PDFs (Camp 2007, Vogt 1940) | Restricted |
| `data/raw/camp_raw.txt` | Extracted text | Restricted |
| `data/processed/*.jsonl` | Training pairs | Restricted |
| `models/nllb-200-600M/` | Base model weights | Public (Meta AI release) |
| `outputs/checkpoints/` | LoRA adapter weights | Restricted — tribal property |
| `mlruns/` | MLflow experiment tracking | Restricted |

---

## Technical Sovereignty Enforcement

Sovereignty is enforced structurally in code, not merely as policy:

### Prohibited at the import level
The following packages must never appear in any source file:
- `boto3`, `s3fs` — AWS
- `google-cloud-*`, `google.cloud.*` — GCP
- `azure-*` — Azure
- `openai` — OpenAI API
- `anthropic` — Anthropic API
- `wandb` — Weights & Biases (remote tracking)
- Any cloud storage or remote inference SDK

### Enforced at runtime
- `local_files_only=True` — all `from_pretrained()` calls cannot reach Hugging Face Hub
- `report_to=[]` — training arguments disable all external experiment reporters
- No `requests.post()` calls to non-localhost URLs in training or inference paths

### Enforced at the repository level
- `data/`, `models/`, `outputs/`, `mlruns/` all listed in `.gitignore`
- Pre-commit audit script verifies no restricted data or prohibited packages are staged before any commit

---

## Researcher Obligations

As a Kalispel tribal member conducting this research, Isaac Alvarado commits to:

1. Presenting all prototype outputs to the Language Director (JR Bluff) before
   any academic publication or public demonstration.
2. Obtaining explicit approval before sharing model weights, training data,
   or evaluation outputs in any form outside the tribe.
3. Treating model errors and low-quality outputs with appropriate epistemic
   humility — never presenting MT output as authoritative Kalispel.
4. Maintaining this governance document as a living record, updated as the
   project evolves and as tribal guidance is received.

---

## Approved Data Sources

| Source | Status | Use |
|--------|--------|-----|
| Camp, K. (2007). *An Interlinear Analysis of Seven Kalispel Texts.* | Approved for prototype | Primary training data |
| Vogt, H. (1940). *The Kalispel Language.* | Pending review | Potential supplementary |
| Giorda, J. (1877). *A Dictionary of the Kalispel Language.* | Pending review | Potential supplementary |
| Tribal curriculum materials | Requires Language Director approval | Future phases only |

---

## Contact

**Language Director:** JR [Last name omitted from public repository]
**Kalispel Tribe of Indians:** Usk, Washington
**Researcher:** Isaac Alvarado — [ASU contact via project PI]

*This document must not be modified without human review.*
*Changes require acknowledgment from the Language Director or designated tribal authority.*

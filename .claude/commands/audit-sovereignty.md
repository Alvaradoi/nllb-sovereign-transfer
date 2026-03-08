# /audit-sovereignty

Sovereignty compliance check. Run before any git operation.

## Steps

1. Scan the git staging area for prohibited files:
   ```bash
   git diff --cached --name-only
   ```
   FAIL if any staged file matches: `data/*`, `models/*`, `outputs/*`, `mlruns/*`, `CLAUDE.md`

2. Check `requirements.txt` for prohibited packages:
   Prohibited: `boto3`, `google-cloud`, `azure`, `openai`, `anthropic`, `wandb`,
   any cloud storage or inference SDK.
   ```bash
   grep -iE "boto3|google-cloud|azure|openai|anthropic|wandb" requirements.txt
   ```
   FAIL if any match found.

3. Scan `src/` for outbound network calls:
   ```bash
   grep -rn "requests.post\|httpx.post\|urllib.request.urlopen" src/
   ```
   FAIL if any call targets a non-localhost URL.

4. Scan `src/` for prohibited imports:
   ```bash
   grep -rn "import boto3\|import openai\|import anthropic\|import wandb\|from google.cloud\|from azure" src/
   ```
   FAIL if any match found.

5. Verify `.gitignore` covers all sovereign paths:
   Confirm entries exist for: `data/`, `models/`, `outputs/`, `mlruns/`, `CLAUDE.md`

## Report

Print PASS or FAIL for each check with details on any failures.
Do not proceed with `git push` if any check fails.

# /run-eval

Evaluate the latest checkpoint against the dev set and log results to MLflow.

## Steps

1. Locate the latest checkpoint:
   ```bash
   ls -t outputs/checkpoints/ | head -1
   ```

2. Run evaluation against the dev split:
   ```bash
   bash scripts/evaluate.sh --checkpoint outputs/checkpoints/latest/ --split dev
   ```

3. Log BLEU and ChrF metrics to local MLflow:
   ```bash
   mlflow ui --backend-store-uri mlruns/
   ```

4. Print a summary including:
   - Checkpoint path used
   - BLEU score (sacrebleu)
   - ChrF score
   - Number of dev pairs evaluated
   - Any translation pairs with notably low scores (for qualitative review)

## Notes
- Evaluation is Salish -> English direction only for reliable automated metrics
- English -> Salish outputs must be flagged for expert (Jessie) review
- All metrics stay local in `mlruns/` — never uploaded externally

# /clean-data

Run the full extraction and cleaning pipeline and report on output quality.

## Steps

1. Run `extract_pdf.py` on all PDFs in `data/raw/`:
   ```bash
   python src/extract_pdf.py --input data/raw/Camp2007.pdf \
       --output data/raw/camp_raw.txt --start-page 15 --audit
   ```

2. Run `data_cleaner.py` to produce JSONL pairs:
   ```bash
   python src/data_cleaner.py --input data/raw/ --output data/processed/
   ```

3. Report the following:
   - Total sentence pairs extracted (train / dev / test split counts)
   - Number of pairs rejected and top rejection reasons
   - Any unknown PUA or CID characters encountered (from audit output)
   - Any Salish text that failed NFC normalization check
   - Path and line count of `data/processed/rejection_log.jsonl`

4. Flag any characters from the critical set that may have been corrupted:
   `ʷ ʔ ɬ ə č š ƛ ʼ m̓ n̓ l̓ y̓ w̓`

## Sovereignty check
- Confirm no files outside `data/processed/` were written
- Confirm no network calls were made during the run

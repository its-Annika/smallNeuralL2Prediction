#!/bin/bash
# Find out-of-vocabulary (OOV) words from a single transcript TSV file using an MFA dictionary

# === CONFIG ===
TRANSCRIPT_TSV="/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69/allTranscripts.tsv"  # your TSV transcript file
DICT_PATH="/fs/nexus-scratch/ashankwi/phonProjectF25/Catalan_MFA_Bundle/cat_dictionary.dict"  # Catalan dictionary
OUTPUT_FILE="/fs/nexus-scratch/ashankwi/phonProjectF25/oov_words.tsv"

# === PREP ===
echo "Scanning transcript TSV: $TRANSCRIPT_TSV"
echo "Using dictionary: $DICT_PATH"
echo "Output will be saved to: $OUTPUT_FILE"
echo

# Extract vocabulary from dictionary (first column, lowercase)
cut -d ' ' -f 1 "$DICT_PATH" | tr '[:upper:]' '[:lower:]' | sort | uniq > /tmp/dict_words.txt

# Extract words from TSV (skip first column, normalize)
cut -f2 "$TRANSCRIPT_TSV" | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/[^a-zàèéíòóúüïç ]//g' | \
    tr -s ' ' '\n' | \
    grep -v '^$' | \
    sort | uniq > /tmp/all_transcript_words.txt

# Compare and find OOVs
comm -23 /tmp/all_transcript_words.txt /tmp/dict_words.txt > "$OUTPUT_FILE"

# === REPORT ===
OOV_COUNT=$(wc -l < "$OUTPUT_FILE")
echo "✅ Found $OOV_COUNT out-of-vocabulary words."
echo "They’ve been saved to: $OUTPUT_FILE"

# Optional: show the first few
echo
echo "Examples:"
head "$OUTPUT_FILE"

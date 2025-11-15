#!/bin/bash

# === CONFIG ===
TRANSCRIPT_FILE="/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69/line_index_male.tsv"      # your master transcript file
AUDIO_ROOT="/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69/ca_es_male"               # root directory with nested audio
OUTPUT_DIR="/fs/nexus-scratch/ashankwi/phonProjectF25/catalan_slr69_textGrids/ca_es_male"          # where TextGrids will be saved
DICT="/fs/nexus-scratch/ashankwi/phonProjectF25/Catalan_MFA_Bundle/cat_complete_dictionary.dict"                   # MFA model (must be downloaded)
MODEL="/fs/nexus-scratch/ashankwi/phonProjectF25/Catalan_MFA_Bundle/cat_acoustic.zip"
NUM_JOBS=8                                # number of parallel jobs for MFA

mkdir -p "$OUTPUT_DIR"
TMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TMP_DIR"

# --- Pre-index all audio files ---
echo "Indexing audio files..."
find "$AUDIO_ROOT" -type f \( -iname "*.flac" -o -iname "*.wav" \) > "$TMP_DIR/all_audio.txt"
echo "Found $(wc -l < "$TMP_DIR/all_audio.txt") audio files."

# --- Count total transcript lines ---
TOTAL_LINES=$(wc -l < "$TRANSCRIPT_FILE")
CURRENT_LINE=0

# --- Process transcript ---
while IFS=$'\t' read -r basename text; do
    ((CURRENT_LINE++))

    # --- Show progress bar ---
    PERCENT=$(( CURRENT_LINE * 100 / TOTAL_LINES ))
    BAR_WIDTH=50
    FILLED=$(( PERCENT * BAR_WIDTH / 100 ))
    EMPTY=$(( BAR_WIDTH - FILLED ))
    BAR=$(printf "%0.s#" $(seq 1 $FILLED))
    BAR="$BAR$(printf "%0.s-" $(seq 1 $EMPTY))"
    printf "\rProcessing %d/%d [%s] %d%%" "$CURRENT_LINE" "$TOTAL_LINES" "$BAR" "$PERCENT"

    # --- Find audio path from pre-indexed list ---
    WAV_PATH=$(grep -i "${basename}" "$TMP_DIR/all_audio.txt" | head -n 1)
    if [ -z "$WAV_PATH" ]; then
        echo -e "\nWARNING: Audio not found for $basename"
        continue
    fi

    # --- Convert to 16 kHz mono WAV in temp folder ---
    TMP_WAV="$TMP_DIR/$basename.wav"
    ffmpeg -loglevel error -y -i "$WAV_PATH" -ar 16000 -ac 1 "$TMP_WAV"

    # --- Normalize transcript ---
    norm=$(echo "$text" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-záéíóúñü ]//g' | tr -s ' ')
    echo "$norm" > "$TMP_DIR/$basename.txt"

done < "$TRANSCRIPT_FILE"

echo -e "\nAll transcripts processed. Starting MFA alignment..."

# --- Run MFA alignment on the entire temp folder ---
mfa align "$TMP_DIR" "$DICT" "$MODEL" "$OUTPUT_DIR" --num_jobs "$NUM_JOBS"

echo "Alignment complete. TextGrids are in: $OUTPUT_DIR"

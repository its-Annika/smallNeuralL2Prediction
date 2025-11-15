#!/bin/bash
# Organize flat TextGrid files into LibriSpeech-style directories
# Example: 133_1378_000000.TextGrid -> 133/1378/133_1378_000000.TextGrid

# === CONFIG ===
SRC_DIR="/fs/nexus-scratch/ashankwi/phonProjectF25/spanish_textGrids/train"      # flat source folder
DEST_DIR="/fs/nexus-scratch/ashankwi/phonProjectF25/realspanish_textGrids/train"  # nested destination folder
MODE="copy"   # use "move" to move instead of copy

# Create destination root if it doesn't exist
mkdir -p "$DEST_DIR"

# Count total files for progress display
total=$(find "$SRC_DIR" -type f -name "*.TextGrid" | wc -l)
count=0

echo "Found $total TextGrid files. Organizing into LibriSpeech-style directories..."

# Process each TextGrid file safely (handles spaces and special chars)
find "$SRC_DIR" -type f -name "*.TextGrid" -print0 | while IFS= read -r -d '' file; do
    ((count++))
    fname=$(basename "$file")

    # Split filename into components: e.g. 133_1378_000000.TextGrid -> 133 / 1378
    IFS="_" read -r spk chap _ <<< "$fname"

    # Construct destination path (preserve full filename)
    outdir="$DEST_DIR/$spk/$chap"
    mkdir -p "$outdir"
    dest="$outdir/$fname"

    # Copy or move depending on MODE
    if [ "$MODE" = "move" ]; then
        mv "$file" "$dest"
    else
        cp -n "$file" "$dest"  # -n = no overwrite
    fi

    # Inline progress indicator
    echo -ne "\r[$count/$total] Placed $fname                         "
done

echo -e "\n Done organizing $total TextGrids into LibriSpeech-style directories."

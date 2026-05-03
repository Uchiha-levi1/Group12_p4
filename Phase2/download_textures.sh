#!/bin/bash
# Download floor texture images in ranked order.
# Ranking:
#   1. Where's Waldo (highest)
#   2. Paolo
#   3. Austin
#   4. Sander (untested)
#   5. Jon (untested)

set -e

OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)/static"

echo "Downloading floor texture images to: $OUTPUT_DIR"
echo ""

# Print rankings
echo "=== Texture Rankings ==="
echo "  1. Where's Waldo Space Station [highest]"
echo "  2. Paolo Bendandi"
echo "  3. Austin Scherbarth"
echo "  4. Sander Crombach [untested]"
echo "  5. Jon Stebbe [untested]"
echo ""

# --- 1. Where's Waldo - Space Station (HIGHEST PRIORITY) ---
echo "[1/5] Downloading Where's Waldo Space Station..."
curl -L -o "$OUTPUT_DIR/Wheres-Waldo-Space-Station-Super-High-Resolution-scaled.jpg" \
  "https://vuss.io/wp-content/uploads/2025/01/Wheres-Waldo-Space-Station-Super-High-Resolution-scaled.jpg"
echo "  Done."

# --- 2. Paolo Bendandi ---
echo "[2/5] Downloading Paolo Bendandi texture..."
curl -L -o "$OUTPUT_DIR/paolo-bendandi-Yj5r_KKh8WU-unsplash.jpg" \
  "https://images.unsplash.com/photo-1526590847572-533c6ae9f542?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&dl=paolo-bendandi-Yj5r_KKh8WU-unsplash.jpg"
echo "  Done."

# --- 3. Austin Scherbarth ---
echo "[3/5] Downloading Austin Scherbarth texture..."
curl -L -o "$OUTPUT_DIR/austin-scherbarth-qSrFTyh-IB0-unsplash.jpg" \
  "https://images.unsplash.com/photo-1486671736870-2f695ecdf813?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&dl=austin-scherbarth-qSrFTyh-IB0-unsplash.jpg"
echo "  Done."

# --- 4. Sander Crombach (untested) ---
echo "[4/5] Downloading Sander Crombach texture (untested)..."
curl -L -o "$OUTPUT_DIR/sander-crombach-Zst8CE0bD9c-unsplash.jpg" \
  "https://images.unsplash.com/photo-1552158300-22b14daea1b1?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&dl=sander-crombach-Zst8CE0bD9c-unsplash.jpg"
echo "  Done."

# --- 5. Jon Stebbe (untested) ---
echo "[5/5] Downloading Jon Stebbe texture (untested)..."
curl -L -o "$OUTPUT_DIR/jon-stebbe-paydk0JcIOQ-unsplash.jpg" \
  "https://images.unsplash.com/photo-1585412727339-54e4bae3bbf9?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&dl=jon-stebbe-paydk0JcIOQ-unsplash.jpg"
echo "  Done."

echo ""
echo "All downloads complete!"

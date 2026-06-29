"""
rebuild_dict_local.py — Rebuild the dictionary from local nationwide cache instantly.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from build_osm_dict
from scripts.build_osm_dict import (
    build_word_frequency,
    build_dictionary,
    clean_and_patch_dict,
    OUTPUT_PATH
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("=== Rebuilding Dictionary Locally from Cache ===")
    cache_path = OUTPUT_PATH.parent / "osm_raw_cache_nationwide.json"
    
    if not cache_path.exists():
        print(f"Error: Cache file {cache_path} does not exist!")
        sys.exit(1)
        
    with open(cache_path, 'r', encoding='utf-8') as f:
        all_names = json.load(f)
        
    print(f"Loaded {len(all_names)} name strings from cache.")
    
    # Build frequency tables
    print("Building nationwide word frequency table...")
    word_freq, bigram_freq = build_word_frequency(all_names)
    print(f"  → {len(word_freq)} unique base words")
    print(f"  → {len(bigram_freq)} unique base bigrams")

    # Build dictionary
    print("\nBuilding canonical dictionary...")
    dictionary = build_dictionary(word_freq, bigram_freq, min_word_count=2, min_bigram_count=3)
    
    # Clean and patch
    dictionary = clean_and_patch_dict(dictionary)
    print(f"  → Final dictionary has {len(dictionary)} entries.")

    # Save dictionary
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\n✓ Dictionary saved to {OUTPUT_PATH}")

    # Verify so
    print(f"Check 'so' in dictionary: {'so' in dictionary}")

if __name__ == '__main__':
    main()

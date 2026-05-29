import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import extract_pois_from_screenshot

# Setup basic logging to see everything
logging.basicConfig(level=logging.DEBUG)

async def test():
    screenshot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots", "tile_417537_246351.png")
    if not os.path.exists(screenshot_path):
        print(f"Error: Screenshot not found at {screenshot_path}")
        return
        
    print(f"Reading screenshot: {screenshot_path}")
    with open(screenshot_path, "rb") as f:
        img_bytes = f.read()
        
    print("Calling vision model...")
    # Import inside to print raw response
    import vision
    original_parse = vision._parse_poi_response
    
    raw_response_holder = []
    
    def debug_parse(text):
        print("\n=== RAW LLM RESPONSE ===")
        print(text)
        print("========================\n")
        raw_response_holder.append(text)
        return original_parse(text)
        
    vision._parse_poi_response = debug_parse
    
    pois, outside = await extract_pois_from_screenshot(img_bytes)
    
    print("\n=== PARSED RESULTS ===")
    print(f"Outside: {outside}")
    print(f"POIs found ({len(pois)}):")
    for p in pois:
        print(f"  - {p}")

if __name__ == "__main__":
    asyncio.run(test())

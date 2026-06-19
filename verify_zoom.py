import asyncio
import re
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to zoom 21...")
        await page.goto('https://www.google.com/maps/@10.779856,106.698557,21z', wait_until='load')
        # Wait a bit for the redirect / zoom settlement
        await page.wait_for_timeout(5000)
        url = page.url
        print(f"Loaded URL: {url}")
        
        match = re.search(r"@[-0-9.]+,[-0-9.]+,([0-9.]+)z", url)
        if match:
            print(f"Parsed Zoom Level: {match.group(1)}")
        else:
            print("Zoom level not found in URL")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test())

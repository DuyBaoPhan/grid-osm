import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Create a context with bypass_csp=True
        context = await browser.new_context(bypass_csp=True)
        page = await context.new_page()
        await page.goto("https://www.openstreetmap.org/#map=19/10.779247/106.700031", wait_until="domcontentloaded")
        
        print("--- Testing style injection with bypass_csp=True ---")
        css = """
        #header, #sidebar, .welcome, .banner, #banner, .announcement, .flash-wrap, #flash, .cookie-consent, #cookie-consent {
            display: none !important;
        }
        #map {
            left: 0 !important;
            top: 0 !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            position: absolute !important;
        }
        """
        try:
            await page.add_style_tag(content=css)
            sidebar_display = await page.evaluate("() => window.getComputedStyle(document.getElementById('sidebar')).display")
            print(f"Style injected successfully! Sidebar display: {sidebar_display}")
        except Exception as exc:
            print("Failed to inject style even with bypass_csp=True:", exc)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

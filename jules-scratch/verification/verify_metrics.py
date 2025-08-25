from playwright.sync_api import sync_playwright, Page, expect

def verify_training_metrics():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the local built file
        page.goto("file:///app/frontend/chimera_frontend/build/index.html")

        # Take a screenshot immediately to see the initial state
        page.screenshot(path="/app/jules-scratch/verification/verification.png")

        browser.close()

if __name__ == "__main__":
    verify_training_metrics()

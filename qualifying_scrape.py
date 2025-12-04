import pandas as pd
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def clean_driver_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"\b[A-Z]{2,4}$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def scrape_vegas_qualifying(headless=True) -> pd.DataFrame:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto("https://www.formula1.com/")
        page.wait_for_timeout(1500)

        try:
            iframe = page.frame_locator("iframe[title='SP Consent Message']")
            iframe.get_by_role("button", name="Reject Non-Essential Cookies").click()
            page.wait_for_timeout(900)
        except:
            pass

        page.get_by_role("link", name="Results Chevron Dropdown").click()
        page.wait_for_timeout(900)
        page.get_by_role("button", name="All").click()
        page.wait_for_timeout(900)

        page.get_by_label("All").get_by_role(
            "link", name="Flag of United States of America Las Vegas"
        ).click()
        page.wait_for_timeout(2000)

        page.get_by_role("button", name="Race Result").click()
        page.wait_for_timeout(2000)

        page.get_by_role("link", name="Qualifying").click()
        page.wait_for_timeout(2000)

        html = page.inner_html("#results-table")
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    headers = [th.get_text(strip=True) for th in soup.select("thead th")]

    rows = []
    for tr in soup.select("tbody tr"):
        tds = tr.select("td")
        row = {}

        for i, h in enumerate(headers):
            if i >= len(tds):
                row[h] = ""
                continue
            cell = tds[i]
            text = cell.get_text(strip=True)
            if "driver" in h.lower():
                row[h] = clean_driver_name(text)
            else:
                row[h] = text

        rows.append(row)

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv("qualifying_results_Vegas.csv", index=False)
    print("Saved qualifying_results_Vegas.csv")
    return df

if __name__ == "__main__":
    df = scrape_vegas_qualifying(headless=False)
    print(df)
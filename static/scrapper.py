# pip install playwright markdown fpdf
# playwright install chromium

from playwright.sync_api import sync_playwright
import os
import time
from urllib.parse import urlparse
import markdown
from fpdf import FPDF

# --- Configuration ---
URLS_TO_SCRAPE = [
    # Main pages
    "https://www.jio.com/",
    "https://www.jio.com/dashboard/",
    
    # Mobile Services
    "https://www.jio.com/mobile",
    "https://www.jio.com/selfcare/plans/mobility/prepaid-plans-home/",
    "https://www.jio.com/selfcare/plans/mobility/postpaid-plans-home/",
    "https://www.jio.com/5g",
    "https://www.jio.com/international-services/",
    "https://www.jio.com/jcms/esim/",
    "https://www.jio.com/selfcare/recharge/mobility/?entrysource=Mobilepage%20header",
    "https://www.jio.com/selfcare/paybill/mobility/",
    "https://www.jio.com/selfcare/interest/sim/",
    
    # JioHome / Fiber / AirFiber
    "https://www.jio.com/jiohome/",
    "https://www.jio.com/airfiber/",
    "https://www.jio.com/selfcare/interest/airfiber/",
    "https://www.jio.com/selfcare/interest/fiber/",
    "https://www.jio.com/selfcare/plans/fiber/fiber-prepaid-plans-home/",
    "https://www.jio.com/selfcare/plans/fiber/fiber-postpaid-plans-list/",
    "https://www.jio.com/selfcare/paybill/fiber/",
    "https://www.jio.com/selfcare/recharge/fiber/?entrysource=Fiberpage%20header",
    
    # Devices & Products
    "https://www.jio.com/devices/",
    "https://www.jio.com/jiopc",
    "https://www.jio.com/jiorouter/",
    "https://www.jio.com/jiobook/technical-specifications/",
    
    # Business
    "https://www.jio.com/business/",
    "https://www.jio.com/business/resources/",
    "https://www.jio.com/business/contact-us",
    
    # Apps
    "https://www.jio.com/apps/",
    "https://www.jio.com/jcms/apps/jiopages",
    
    # Support & Services
    "https://www.jio.com/help/home/",
    "https://www.jio.com/help/contact-us",
    "https://www.jio.com/track-order",
    "https://www.jio.com/selfcare/track-orders/",
    "https://www.jio.com/selfcare/login/",
]


OUTDIR_MD = "scraped_documents_md"
OUTDIR_PDF = "scraped_documents_pdf"
os.makedirs(OUTDIR_MD, exist_ok=True)
os.makedirs(OUTDIR_PDF, exist_ok=True)

# --- Functions ---

def save_md(url, title, body):
    """Save page content as markdown file"""
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "_") or "home"
    filename = f"{OUTDIR_MD}/{parsed.netloc}__{slug}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Title: {title}\n")
        f.write(f"# Source: {url}\n")
        f.write(f"# Scraped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(body)
    
    print("Saved MD:", filename)
    return filename

def md_to_pdf_fpdf(md_file):
    """Convert markdown file to PDF using FPDF"""
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to HTML text (basic conversion)
    html_text = markdown.markdown(md_text)
    lines = html_text.splitlines()

    pdf_filename = f"{OUTDIR_PDF}/{os.path.basename(md_file).replace('.md','.pdf')}"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in lines:
        pdf.multi_cell(0, 8, line)
    
    pdf.output(pdf_filename)
    print("Saved PDF:", pdf_filename)

# --- Main Scraping ---
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for url in URLS_TO_SCRAPE:
        try:
            page.goto(url, timeout=30000)
            time.sleep(3)  # wait for JS to load

            title = page.title()
            body = page.inner_text("body")  # all visible text

            # Save MD
            md_file = save_md(url, title, body)

            # Convert to PDF
            md_to_pdf_fpdf(md_file)

            time.sleep(2)  # polite delay
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            time.sleep(5)

    browser.close()

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from pathlib import Path

# Silencing Warnings via Logging
import logging
logging.getLogger("fitz").setLevel(logging.ERROR)

def read_source(source, source_type):
    """
    Reads and extracts text from different source types.

    Args:
        source (str): The path or URL of the source.
        source_type (str): Type of source. Options: 'local_pdf', 'online_pdf', 'website'

    Returns:
        str: Extracted text content.
    """
    if source_type == "local_pdf":
        return read_local_pdf(source)

    elif source_type == "online_pdf":
        return read_online_pdf(source)

    elif source_type == "website":
        return read_website(source)

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    


# Helper functions

def read_local_pdf(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Local PDF not found at {file_path}")
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_online_pdf(url):
    response = requests.get(url)
    response.raise_for_status()
    temp_path = Path("temp_online.pdf")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    text = read_local_pdf(temp_path)
    temp_path.unlink()  # Clean up temp file
    return text

def read_website(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Removing script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    return text


# =====================
# Example Test Run
# =====================
if __name__ == "__main__":
    # Example 1: Local PDF test
    try:
        local_pdf_path = "Report7 - Amazon 2024.pdf"
        text_pdf_local = read_source(local_pdf_path, "local_pdf")
        print("\n✅ Local PDF Sample Text:\n", text_pdf_local[:500])
    except Exception as e:
        print(f"❌ Local PDF Test Failed: {e}")

    # Example 2: Online PDF test
    # Replace with a valid PDF URL for testing
    try:
        sample_pdf_url = "https://sustainability.aboutamazon.com/2024-amazon-sustainability-report.pdf"
        text_pdf_online = read_source(sample_pdf_url, "online_pdf")
        print("\n✅ Online PDF Sample Text:\n", text_pdf_online[:500])
    except Exception as e:
        print(f"❌ Online PDF Test Failed: {e}")

    # Example 3: Website test
    try:
        website_url = "https://sustainability.aboutamazon.com/2024-report"
        text_website = read_source(website_url, "website")
        print("\n✅ Website Sample Text:\n", text_website[:500])
    except Exception as e:
        print(f"❌ Website Test Failed: {e}")
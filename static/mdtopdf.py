import pypandoc
import os

def md_to_pdf(input_path, output_path=None):
    """
    Convert a Markdown (.md) file to PDF using Pandoc + XeLaTeX.
    Handles Unicode characters (₹, emojis, etc.).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # Make sure pandoc is installed
    pypandoc.download_pandoc()

    # Read markdown content (UTF-8 safe)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Optional: Replace problematic symbols if needed
    # text = text.replace("₹", "Rs.")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"

    try:
        # Use XeLaTeX for Unicode support
        pypandoc.convert_text(
            text,
            'pdf',
            format='md',
            outputfile=output_path,
            extra_args=['--standalone', '--pdf-engine=xelatex']
        )
        print(f"✅ Converted: {input_path} → {output_path}")

    except RuntimeError as e:
        print(f"⚠️ Conversion failed for {input_path}: {e}")
        print("Trying fallback to HTML-based conversion...")

        # Fallback to HTML-based PDF if XeLaTeX not found
        fallback_path = output_path.replace(".pdf", "_fallback.pdf")
        pypandoc.convert_text(
            text,
            'pdf',
            format='md',
            outputfile=fallback_path,
            extra_args=['--standalone', '--pdf-engine=weasyprint']
        )
        print(f"✅ Fallback conversion done: {fallback_path}")


if __name__ == "__main__":
    input_folder = "scraped_documents_md"   # Folder with .md files
    output_folder = "scraped_documents_pdf"  # Output folder for PDFs

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".md"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".pdf")
            md_to_pdf(input_file, output_file)

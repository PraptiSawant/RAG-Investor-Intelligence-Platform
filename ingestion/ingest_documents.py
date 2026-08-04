import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from rag.kpi_extractor_rag import extract_financial_metrics, Retriever
from database.save_metrics import save_metrics
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    """Parse company and year from a PDF filename.

    Supports names like `2024_Apple.pdf` and `2024_AnnualReport_Apple.pdf`.
    """
    stem = pdf_file.stem
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        year = parts[0]
        company = parts[-1]
    elif len(parts) >= 2:
        company = parts[0]
        year = parts[1]
    else:
        company = stem
        year = ""

    return company, year


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store
) -> None:
    """
    Ingest a single PDF document.
    """
    pdf_file = Path(pdf_path)

    company, year = parse_company_year(pdf_file)
    print(f"Ingesting {pdf_file.name} as company={company!r}, year={year!r}")

    converter = PDFToMarkdownConverter()

    markdown_file = converter.convert_pdf(
        pdf_path=pdf_path,
        output_dir="data/markdown"
    )

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks for {pdf_file.name}")


    cleaned_year = int(year) if year.isdigit() else None
    for chunk in chunks:
        chunk.metadata.update({
            "company": company,
            "year": cleaned_year,
            "source_file": pdf_file.name
        })

    vector_store.add_documents(documents=chunks)
    print(f"Successfully uploaded chunks to Pinecone for {pdf_file.name}!")

    # Extract financial metrics using the newly ingested data
    metrics = extract_financial_metrics(
        retriever=Retriever(vector_store=vector_store),
        company=company,
        year=cleaned_year
    )

    # Persist metrics to PostgreSQL
    if metrics:
        save_metrics(company=company, year=cleaned_year, metrics=metrics)


def ingest_directory(input_dir: str) -> None:
    """
    Ingest all PDFs from a directory.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = PineconeVectorStore(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            vector_store=vector_store
        )


if __name__ == "__main__":
    ingest_directory("data/raw_pdfs")
    # ingest_document("data/raw_pdfs/2024_Apple.pdf")
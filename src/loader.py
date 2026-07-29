from pathlib import Path
from bs4 import BeautifulSoup
import fitz
import markdown
import json
from typing import List, Dict

class DocumentLoader:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        print("Raw Directory:", self.raw_dir.resolve())
        print("Processed Directory:", self.processed_dir.resolve())

    def load_txt(self, filepath: Path) -> List[Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return [{
            "text":text.strip(),
            "metadata":{
                "source": filepath.name,
                "page" : None,
                "section" : None,
                "type" : "txt"
            }
        }]

    def load_pdf(self, filepath: Path) -> List[Dict]:
        pdf = fitz.open(filepath)
        documents = []
        for page_no, page in enumerate(pdf):
            text = page.get_text()
            documents.append({
                "text": text.strip(),
                "metadata":{
                    "source": filepath.name,
                    "page": page_no +1,
                    "section": None,
                    "type" : "pdf"
                }
            })
        return documents

    def load_markdown(self, filepath : Path) -> List[Dict]:
        with open(filepath, "r", encoding = "utf-8") as f:
            text = f.read()
        html = markdown.markdown(text)
        txt = BeautifulSoup(html, "html.parser").get_text("\n")
        return [{
            "text" : txt.strip(),
            "metadata": {
                "source" : filepath.name,
                "page" : None,
                "section" : None,
                "type" : "md"
            }
        }]
    
    def load_html(self, filepath: Path) -> List[Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        text = soup.get_text(separator= "\n")
        return [{
            "text" : text.strip(),
            "metadata": {
                "source" : filepath.name,
                "page" : None,
                "section" : None,
                "type" : "html"
            }
        }]

    def load_document(self, filepath: Path):
        suffix = filepath.suffix.lower()
        if suffix == ".txt":
            return self.load_txt(filepath)
        elif suffix == ".pdf":
            return self.load_pdf(filepath)
        elif suffix == ".md":
            return self.load_markdown(filepath)
        elif suffix == ".html":
            return self.load_html(filepath)
        else:
            print(f"Unsupported file type: {suffix}")
            return []

    def save_processed(self, documents, filename):
        output = self.processed_dir / f"{filename}.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent = 4, ensure_ascii=False)
        print(f"Processed documents saved to {output}")

    def process_all(self):
        for file in self.raw_dir.rglob("*"):
            print(f"Processing {file.name}")
            docs = self.load_document(file)
            self.save_processed(docs, file.stem)
        print("Finished")
        
loader = DocumentLoader(raw_dir = "data/raw", processed_dir = "data/processed")
loader.process_all()
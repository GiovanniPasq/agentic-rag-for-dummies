from pathlib import Path
import shutil
import config
from util import pdfs_to_markdowns

class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
    def add_documents(self, pdf_paths, progress_callback=None):
        if not pdf_paths:
            return 0, 0
            
        pdf_paths = [pdf_paths] if isinstance(pdf_paths, str) else pdf_paths
        pdf_paths = [p for p in pdf_paths if p and Path(p).suffix.lower() == ".pdf"]
        
        if not pdf_paths:
            return 0, 0
            
        added = 0
        skipped = 0
        
        for i, pdf_path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback((i + 1) / len(pdf_paths), f"Processing {Path(pdf_path).name}")
                
            pdf_name = Path(pdf_path).stem
            md_path = self.markdown_dir / f"{pdf_name}.md"
            
            if md_path.exists():
                skipped += 1
                continue
                
            try:
                pdfs_to_markdowns(str(pdf_path), overwrite=False)                
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
                
                if not child_chunks:
                    skipped += 1
                    continue
                
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)
                
                added += 1
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                skipped += 1
            
        return added, skipped
    
    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])
    
    def clear_all(self):
        if self.markdown_dir.exists():
            shutil.rmtree(self.markdown_dir)
            self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
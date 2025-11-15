import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChuncker:
    def __init__(self, min_parent_size_multiplier=config.PARENT_SIZE_MUTIPLIER):
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE, 
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.CHILD_CHUNK_SIZE * min_parent_size_multiplier

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)
        
        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())
        
        merged_parents = self.__merge_small_parents(parent_chunks)
        
        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, merged_parents, doc_path)
        
        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks):
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        
        if current:
            merged.append(current)
        
        return merged

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": str(doc_path.stem)+".pdf", "parent_id": parent_id})
            
            all_parent_pairs.append((parent_id, p_chunk))
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))
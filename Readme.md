# Multimodal RAG Tutorial (Text + Images)

## Introduction

Multimodal Retrieval-Augmented Generation (RAG) extends traditional RAG
by enabling retrieval across multiple modalities such as text and
images. This tutorial explains how to build a Multimodal RAG system
using:

-   OpenAI models
-   LangChain
-   ChromaDB
-   PyMuPDF
-   Pillow

------------------------------------------------------------------------

## 1. Prerequisites

Install required libraries:

``` bash
pip install openai langchain chromadb pymupdf pillow tiktoken
```

Make sure: - Python \>= 3.8 - OpenAI API key is configured - Internet
connection is available

------------------------------------------------------------------------

## 2. Project Structure

    MultiModal-RAG/
    │
    ├── multimodal_sample.pdf
    ├── multimodalopenai.ipynb
    ├── embeddings/
    └── assets/

------------------------------------------------------------------------

## 3. Step 1: Import Libraries

``` python
import fitz  # PyMuPDF
from PIL import Image
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from openai import OpenAI
```

------------------------------------------------------------------------

## 4. Step 2: Extract Text and Images from PDF

``` python
doc = fitz.open("multimodal_sample.pdf")
pages_text = []
images = []

for page_index, page in enumerate(doc):
    text = page.get_text("text")
    pages_text.append(text)

    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:
            img_data = pix.get_image_data(output="png")
            images.append(img_data)
```

------------------------------------------------------------------------

## 5. Step 3: Create Embeddings and Vector Database

``` python
embeddings = OpenAIEmbeddings()

vector_db = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=embeddings
)
```

------------------------------------------------------------------------

## 6. Step 4: Convert Data into Documents

``` python
docs = []

# Add text pages
for idx, text in enumerate(pages_text):
    docs.append(
        Document(
            page_content=text,
            metadata={"page": idx, "type": "text"}
        )
    )

# Add image captions (example placeholder)
for idx, img_data in enumerate(images):
    caption = "Image content description placeholder"
    docs.append(
        Document(
            page_content=caption,
            metadata={"image_index": idx, "type": "image"}
        )
    )

vector_db.add_documents(docs)
```

------------------------------------------------------------------------

## 7. Step 5: Create Retriever

``` python
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
```

------------------------------------------------------------------------

## 8. Step 6: Ask Questions (RAG Pipeline)

``` python
query = "What does the chart on page 3 show?"

relevant_docs = retriever.get_relevant_documents(query)

context = "\n".join([doc.page_content for doc in relevant_docs])

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context}
    ]
)

print(response.choices[0].message.content)
```

------------------------------------------------------------------------

## 9. Best Practices

-   Split long text into smaller chunks
-   Use proper image captioning models (BLIP, GPT-4 Vision)
-   Store metadata carefully
-   Tune retrieval parameter `k`
-   Use persistent Chroma storage for production

------------------------------------------------------------------------

## 10. Optional Enhancements

  Enhancement        Benefit
  ------------------ -----------------------------
  OCR on images      Extract embedded text
  CLIP embeddings    Better multimodal alignment
  LangChain Agents   Advanced reasoning
  Persistent DB      Production-ready system

------------------------------------------------------------------------

## Conclusion

You have now built a complete Multimodal RAG system that:

1.  Extracts text and images from PDFs
2.  Embeds both modalities
3.  Stores them in a vector database
4.  Retrieves relevant context
5.  Generates answers using a language model

This architecture can be extended to research papers, medical reports,
legal documents, and multimodal knowledge systems.

------------------------------------------------------------------------

**Author:** Muddassar Ali\
**Project:** MultiModal-RAG

# Private Rag

A test to do RAG locally.

## Setup

This assumes you're running LMStudio and have downloaded the following models:

* LLM: gemma-3-12b-it-qat
* Text embeddings: text-embedding-nomic-embed-text-v1.5

The example loads, splits and calculate embeddings on the included Pickeball rulebook PDF

It runs a local `gradio` instance to provide a UI to sumbit prompts and receive responses.


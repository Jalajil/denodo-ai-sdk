![Denodo Logo](api/static/denodo_logo.png)

# Denodo AI SDK

The Denodo AI SDK is an open-source component designed to streamline the integration of the Denodo Platform with Large Language Models (LLMs). It provides developers with the essential tools to build high-performance AI agents that can interact natively with enterprise data through a governed data virtualization layer.

By automating the complexities of Retrieval-Augmented Generation (RAG) and VQL query generation, the SDK ensures AI responses are grounded in real-time, factual enterprise context.

## Key features

- **Text-to-VQL**: Automatically translates natural language questions into Denodo Virtual Query Language (VQL), allowing AI agents to query the Denodo Platform directly without manual SQL/VQL coding.
- **Metadata Search (RAG)**: Uses vectorization to index and search through your technical and business metadata. This enables agents to find the right data assets instantly using semantic search.
- **DeepQuery Agent**: A sophisticated research agent that orchestrates complex tasks. It crafts a multi-step execution plan and carries it out by combining metadata discovery with live VQL data extraction.
- **Model & Vector Store Agnostic**: Provides a flexible architecture to configure and switch between various LLMs and Vector Databases depending on your performance and privacy requirements.

To showcase the AI SDK’s capabilities, a sample chatbot application is included.

The complete user manual for the Denodo AI SDK is available [here](https://community.denodo.com/docs/html/document/denodoconnects/latest/en/Denodo%20AI%20SDK%20-%20User%20Manual).

### Installation

To get started with the AI SDK:

1. Clone this repository and `cd` into it
2. Create a new virtual environment (`python -m venv venv`) in the root of the AI SDK's path
3. Activate the virtual environment (`source venv/bin/activate` for Linux/MacOS or `.\venv\Scripts\activate` for Windows)
4. Install the requirements.txt (`python -m pip install -r requirements.txt`)
5. Rename the configuration templates for both AI SDK (`api/utils/sdk_config.env.example` => `api/utils/sdk_config.env`) and the sample chatbot (`sample_chatbot/chatbot_config.env.example` => `sample_chatbot/chatbot_config.env`)
5. Review the configuration files for both the AI SDK and the sample chatbot and configure your own LLM/embeddings providers

## AI SDK Benchmarks

We test our text-to-VQL pipeline on our propietary benchmark across the whole range of LLMs that we support.
The benchmark dataset consists of 50+ questions in the finance sector.
You may use this benchmark as reference to choose an LLM model.

<em>Latest update: 05/18/2026 on AI SDK version 1.2</em>

| LLM Provider | Model | 🎯 Accuracy | 🔢 Input Tokens | 🔡 Output Tokens | 📊 Total Tokens | 💰 Cost per Query |
|--------------|-------|-------------|----------------|-----------------|-----------------|-------------------|
| OpenAI       | gpt-5.5                       | 🟢           |          7,092 |             396 |           7,488 |            $0.047 |
| OpenAI       | gpt-5.5-pro                   | 🟢           |          4,335 |           2,330 |           6,665 |            $0.549 |
| OpenAI       | gpt-5.4                       | 🟢           |          7,825 |             433 |           8,258 |            $0.026 |
| OpenAI       | gpt-5.4-pro                   | 🟢           |          3,967 |           1,643 |           5,610 |            $0.415 |
| OpenAI       | gpt-5.4-nano                  | 🟡           |          8,632 |             509 |           9,141 |            $0.002 |
| OpenAI       | gpt-5.4-mini                  | 🟢           |          6,942 |             394 |           7,336 |            $0.007 |
| OpenAI       | gpt-oss-120b                  | 🟡           |          6,636 |             947 |           7,583 |            $0.001 |
| OpenAI       | gpt-oss-20b                   | 🟢           |          7,098 |           2,172 |           9,270 |            $0.001 |
| OpenAI       | gpt-5.3-codex                 | 🟡           |          7,601 |             446 |           8,047 |            $0.020 |
| Google       | gemini-3.1-pro-preview        | 🟢           |          7,825 |           3,726 |          11,551 |            $0.060 |
| Google       | gemini-3-flash-preview        | 🟢           |          9,397 |             456 |           9,853 |            $0.006 |
| Google       | gemini-3.1-flash-lite-preview | 🟢           |          8,430 |             442 |           8,872 |            $0.003 |
| Google       | gemma-4-31b-it                | 🟢           |          8,527 |             442 |           8,969 |            $0.001 |
| Google       | gemma-4-26b-a4b-it            | 🟡           |          9,103 |             423 |           9,526 |            $0.001 |
| Anthropic    | claude-opus-4.7               | 🟢           |         12,994 |             501 |          13,495 |            $0.077 |
| Anthropic    | claude-sonnet-4.6             | 🟢           |          8,770 |             424 |           9,194 |            $0.033 |
| Anthropic    | claude-haiku-4.5              | 🟢           |          8,856 |             469 |           9,325 |            $0.011 |
| DeepSeek     | deepseek-v4-pro               | 🟡           |          8,856 |             481 |           9,337 |            $0.004 |
| DeepSeek     | deepseek-v4-flash             | 🟡           |          8,492 |             402 |           8,894 |            $0.001 |
| MoonshotAI   | kimi-k2.6                     | 🟡           |          8,068 |             691 |           8,759 |            $0.008 |
| Qwen         | qwen3.6-plus                  | 🟡           |          7,795 |             406 |           8,201 |            $0.003 |
| Qwen         | qwen3.6-27b                   | 🟡           |          8,215 |             587 |           8,802 |            $0.005 |
| Qwen         | qwen3.6-35b-a3b               | 🟢           |          8,356 |             763 |           9,119 |            $0.002 |
| Qwen         | qwen3.6-max-preview           | 🟡           |          8,247 |             476 |           8,723 |            $0.012 |
| Qwen         | qwen3.6-flash                 | 🟡           |          8,238 |             774 |           9,012 |            $0.002 |
| xAI          | grok-4.3                      | 🔴           |          8,358 |             345 |           8,703 |            $0.011 |
| MiniMax      | minimax-m2.7                  | 🟡           |          6,915 |           3,080 |           9,995 |            $0.006 |
| MistralAI    | mistral-small-2603            | 🟡           |          8,637 |             457 |           9,094 |            $0.002 |
| MistralAI    | ministral-14b-2512            | 🟡           |          9,251 |             677 |           9,928 |            $0.002 |
| MistralAI    | mistral-medium-3-5            | 🟡           |          8,889 |             427 |           9,316 |            $0.017 |
| MistralAI    | devstral-2512                 | 🟡           |          8,992 |             424 |           9,416 |            $0.004 |
| Meta         | llama-4-maverick              | 🟡           |          7,620 |             489 |           8,109 |            $0.001 |
| Meta         | llama-4-scout                 | 🟡           |          9,469 |             718 |          10,187 |            $0.001 |
| Z-AI         | glm-5.1                       | 🟡           |          8,724 |             442 |           9,166 |            $0.001 |
| Z-AI         | glm-5-turbo                   | 🟡           |          8,063 |             360 |           8,423 |            $0.011 |

Important things to note:

- "Input Tokens", "Output Tokens", and "Total Tokens" are the average per query.
- Any model with its size in the name, i.e.: gpt-oss-**20b**, represents an **open-source model**.
- Each color corresponds to the following range in terms of accuracy:
  - 🟢 = 90%+
  - 🟡 = 80%–89%
  - 🔴 = <80%

## List of supported LLM providers

The Denodo AI SDK supports the following LLM providers:

* OpenAI
* AzureOpenAI
* Bedrock
* Google
* GoogleAIStudio
* Anthropic
* NVIDIA
* Groq
* Ollama
* Mistral
* SambaNova
* OpenRouter

Where Bedrock refers to AWS Bedrock, NVIDIA refers to NVIDIA NIM and Google refers to Google Vertex AI.

## List of supported embedding providers + recommended

* OpenAI (text-embedding-3-large)
* AzureOpenAI (text-embedding-3-large)
* Bedrock (amazon.titan-embed-text-v2:0)
* Google (text-multilingual-embedding-002)
* Ollama (qwen3-embedding:8b)
* Mistral (mistral-embed)
* NVIDIA (baai/bge-m3)
* GoogleAIStudio (gemini-embedding-exp-03-07)

Where Bedrock refers to AWS Bedrock, NVIDIA refers to NVIDIA NIM and Google refers to Google Vertex AI.

## Data question defaults and sampled rows

Question endpoints default to `mode="data"` where selectable, `verbose=false`,
`enable_query_reviewer=true`, and `vector_search_sample_data_k=8`. After an initial
query returns no rows, the Query Reviewer can make up to three review attempts.
It stops early when a query succeeds or the reviewer keeps the current query.
The Query Fixer retains its two-attempt limit. Requests can override the defaults.
With `verbose=false`, the streaming endpoints return the structured result as
`application/json`; `verbose=true` retains the `text/plain` answer stream.

Each sampled row is stored as a JSON object in `row_json`, with JSON column names
in `columns`. The embedding text pairs column labels with their values. Retrieval
reads the JSON row, and prompts show aligned JSON arrays: values at the same array
index belong to the same sampled row. Empty strings, nulls, zero, and false retain
their positions. Missing cells use an empty string placeholder. Row restrictions
skip sample retrieval, and restricted columns are removed before prompt rendering.

After upgrading, run `/getMetadata` for the relevant databases or tags with
`insert=true` and `incremental=false` to rebuild stored samples and embeddings.
Old comma-separated rows remain readable when their column count matches, but
ambiguous or malformed rows are skipped. Re-synchronization is required to recover
values whose commas were ambiguous in the old format.

## Qdrant configuration

The SDK configuration template selects `VECTOR_STORE=Qdrant`. Configure
`api/utils/sdk_config.env` with `QDRANT_URL` and, if needed, `QDRANT_API_KEY`.
For an embedded instance use `QDRANT_PATH`; `QDRANT_LOCATION=:memory:` is for tests.
The SDK uses the `ai_sdk_vector_store` and `ai_sdk_sample_data` collections.
Hybrid retrieval is enabled by default, combining dense embeddings with BM25.
Set `QDRANT_HYBRID=0` for dense retrieval only. Additional options are documented
in `api/utils/sdk_config.env.example`.
Enabling hybrid on an existing dense collection requires [Qdrant server 1.18+](https://qdrant.tech/documentation/manage-data/collections/#update-vector-schema)
and a full metadata synchronization to populate its new sparse vectors.

Set `LLM_PROVIDER`, `LLM_MODEL`, `THINKING_LLM_PROVIDER`, `THINKING_LLM_MODEL`,
`EMBEDDINGS_PROVIDER`, and `EMBEDDINGS_MODEL` to your deployed services.
The Query Reviewer and Query Fixer use the request's general LLM. Assigning
different models to individual jobs is a separate configuration change.

Run the regression suite with `python -m pytest tests -q` after installing the
project requirements and `pytest`. The tests use in-memory Qdrant and test
embeddings; the BM25 test uses locally cached model files and skips if unavailable.

# Licensing
Please see the file called LICENSE.

# LLM Inference Workshop


## 1. GPU Resources

### GPU Nodes
- **sh03-18n07**: 8 A100/80G GPU, 128 CPU cores, 1000G Memory
- **sh04-06n05**: 4 H100/80G GPU, 64 CPU cores, 1000G Memory
- **marlowe**: 31 nodes, 8 H100/80G GPU per node



## 2. Open Source LLMs

#### Meta Models
- **Llama 3.3 70B** (7B, 13B, 70B): /oak/stanford/groups/maggiori/GCAP/data/llm_models/llama3_3_hf/Llama-3.3-70B-Instruct-AWQ

#### Google Models
- **gemma3 27B** (2.7B): /oak/stanford/groups/maggiori/GCAP/data/llm_models/gemma3/gemma-3-27b-it

#### OpenAI Models
- **GPT OSS 120B**: /oak/stanford/groups/maggiori/GCAP/data/llm_models/openai/gpt-oss-120b
- **GPT OSS 20B**: /oak/stanford/groups/maggiori/GCAP/data/llm_models/openai/gpt-oss-20b


#### Qwen Models
- **Qwen 2.5 72B** /oak/stanford/groups/maggiori/GCAP/data/llm_models/qwen2_5/Qwen2.5-72B-Instruct-AWQ
- **Qwen 3 235B** /oak/stanford/groups/maggiori/GCAP/data/llm_models/qwen3/Qwen3-235B-A22B-AWQ


### Model Download and Setup
```bash
cd $gcap_data/llm_models/qwen3
mkdir -p Qwen3-30B-A3B
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./Qwen3-30B-A3B
```


## 3. AI1 Architecture

### Overview
AI1 is a LLM inference pipeline designed for efficient LLM inference for geoeconomics:



## 4. Build and Run AI1 Pipeline


## 5. Run LLM as API


# LLM Inference Workshop


## 1. GPU Resources

### GPU Nodes
- **sh03-18n07**: 8 A100/80G GPU, 128 CPU cores, 1000G Memory
- **sh04-06n05**: 4 H100/80G GPU, 64 CPU cores, 1000G Memory
- **marlowe**: 31 nodes, 8 H100/80G GPU per node



## 2. Open Source LLMs

#### Meta Models
- **Llama 3.3 70B** : /oak/stanford/groups/maggiori/GCAP/data/llm_models/llama3_3_hf/Llama-3.3-70B-Instruct-AWQ

#### Google Models
- **gemma3 27B** : /oak/stanford/groups/maggiori/GCAP/data/llm_models/gemma3/gemma-3-27b-it

#### OpenAI Models
- **GPT OSS 120B**: /oak/stanford/groups/maggiori/GCAP/data/llm_models/openai/gpt-oss-120b
- **GPT OSS 20B**: /oak/stanford/groups/maggiori/GCAP/data/llm_models/openai/gpt-oss-20b


#### Qwen Models
- **Qwen 2.5 72B** /oak/stanford/groups/maggiori/GCAP/data/llm_models/qwen2_5/Qwen2.5-72B-Instruct-AWQ
- **Qwen 3 235B** /oak/stanford/groups/maggiori/GCAP/data/llm_models/qwen3/Qwen3-235B-A22B-AWQ


### Download models from HuggingFace
```bash
source activate gpu1
cd $gcap_data/llm_models/qwen3
mkdir -p Qwen3-30B-A3B
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./Qwen3-30B-A3B
```


## 3. AI1 Architecture

### Overview
AI1 is a LLM inference pipeline designed for large scale financial document analysis with efficient LLM inference, the main module pipeline is an vLLM-powered DAG (Directed Acyclic Graph) workflow framework for processing financial documents and transcripts.

### Core components
- LlmDAG: Manages workflow execution, supports YAML-based configuration.
- Node: Individual processing units in the pipeline, executes registered transform functions.
- Transform Functions: load_transcripts, analyze_transcripts, analyze_jpm_report, analyze_financial_reports.


## 4. Build and Run AI1 Pipeline

We use Makefile as task automation and build orchestration tool in Python projects. The build system containerizes the LLM pipeline for SLURM execution on Sherlock.

### Build targets:
- **build_ai1**: Builds Apptainer container for the AI1 pipeline.
- **build_vllmapi**: Builds Apptainer container for vLLM API service.

### Run targets:
- **sync_prompts**: Copies prompts fo shared GCAP data directory.
- **run_* targets**: Submit various analysis jobs(transcripts, JPM, Orbit, Fitch) to SLURM with different YAML configs.
- **run_vllm_api**: Starts vLLM API service via SLURM. 

## 5. Steps to process transcripts

### Load transcripts
Run ai1/scratch/xwfeng/load_transcripts.ipynb with date filters and save the transcript parquet file in /oak/stanford/groups/maggiori/GCAP/data/shared/ai_geo1/output/transcripts

### Configure YAML file
- broad.yaml:
```yaml
dag:
  id: "transcript_analysis_dag"
  description: "Transcript Analysis DAG - Broad"

  nodes:
    - id: "analyze_transcripts"
      node_desc: "Analyze transcripts"
      input_files:
        - "/oak/stanford/groups/maggiori/GCAP/data/shared/ai_geo1/temp/transcripts/transcripts_*_marlowe_sample.parquet"
      output_file: "/oak/stanford/groups/maggiori/GCAP/data/ai_geo1/temp/transcripts/broad_analysis_gpt_120b_{}_marlowe_sample_baseline.parquet"
      system_prompt: "/oak/stanford/groups/maggiori/GCAP/data/ai1/prompts/v6/sys_prompt_broad.txt"
      user_prompt: "/oak/stanford/groups/maggiori/GCAP/data/ai1/prompts/v6/user_prompt_broad.txt"
      transform: "analyze_transcripts"
      llm: "llama"
      llm_params:
        model: "/oak/stanford/groups/maggiori/GCAP/data/llm_models/openai/gpt-oss-120b"
        multi_processing: True
        num_gpus: 8
        max_ctx_len: 128000
        temperature: 0
        top_p: 1
        max_tokens: 1024
        seed: 1     
      years: [2022, 2025]
```
- long_*.yaml
```yaml
dag:
  id: "transcript_analysis_dag"
  description: "Transcript Analysis DAG - Long"

  nodes:
    - id: "analyze_transcripts"
      node_desc: "Analyze transcripts"
      input_files:
        - "/oak/stanford/groups/maggiori/GCAP/data/shared/ai_geo1/output/transcripts/llama3/transcripts_*_tariffs.parquet"
      output_file: "/oak/stanford/groups/maggiori/GCAP/data/ai_geo1/temp/transcripts/long_analysis_llama33_70b_{}_experimental_tariffs.parquet"
      system_prompt: "/oak/stanford/groups/maggiori/GCAP/data/ai1/prompts/experimental/sys_prompt_tariff_decreases.txt"
      user_prompt: "/oak/stanford/groups/maggiori/GCAP/data/ai1/prompts/v6/user_prompt_tariff_analysis.txt"
      transform: "analyze_transcripts"
      llm: "llama"
      llm_params:
        model: "/scratch/groups/maggiori/raw_model_weights/Llama-3.3-70B-Instruct-AWQ"
        multi_processing: True
        num_gpus: 8
        max_ctx_len: 128000
        temperature: 0
        top_p: 1
        max_tokens: 2048
        seed: 1     
      years: [2016, 2017, 2023, 2024]
```

### Copy prompts
Run command ```make sync_prompts``` to copy prompts to $gcap_data/ai1/prompts/

### Submit SLURM job
Run command to submit SLURM jobs:
- ```make run_transcript_broad```
- ```make run_transcript_ec```
- ```make run_transcript_sanctions```
- ```make run_transcript_tariffs``` 

Once it's allocated and assigned a job id, run command ```tail -f $gcap_data/ai1/logs/ai1-$jobid.err``` to monitor the progress.


## 6. Run LLM as API Service

### Start the vllm api server
```
make run_vllm_api
```

Follow the Jupyter notebook vllmapi_transcript_analysis.ipynb to call LLM as API.
.PHONY: help install clean predict train apptainer-build apptainer-run

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install dependencies using uv
	uv pip install -e .

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf checkpoints/*.pt
	rm -rf logs/*.log
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Training commands
train: ## Train with single GPU/CPU/MPS
	DEPLOYMENT_ENV=local python ./src/train.py

train-multigpu: ## Train with multiple NVIDIA GPUs (single node, CUDA only)
	DEPLOYMENT_ENV=local torchrun --nproc_per_node=auto --standalone ./src/train.py

# Inference
predict: ## Run single text prediction (usage: make predict TEXT="your text here")
	cd src && python inference.py --text "$(TEXT)"

# Example inference commands
example-predict: ## Run example prediction
	cd src && python inference.py --text "This movie is absolutely fantastic!"

# Testing
test: ## Run model evaluation on test dataset
	cd src && python inference.py --test --model_path ../checkpoints/best_model.pt --test_file ../data/test_data.parquet

# Apptainer commands
apptainer-build: ## Build Apptainer image
	apptainer build containers/text_classifier.sif containers/text_classifier.def

apptainer-run: ## Run Apptainer container
	apptainer run --bind $(PWD)/data:/app/data,$(PWD)/checkpoints:/app/checkpoints containers/text_classifier.sif


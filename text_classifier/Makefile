.PHONY: clean train test predict apptainer-build apptainer-run apptainer-test apptainer-predict gpu-info

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf checkpoints/*.pt
	rm -rf logs/*.log
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Local Training Commands
train: ## Train with single GPU/CPU/MPS (usage: make train [EPOCHS=N])
	DEPLOYMENT_ENV=local $(if $(EPOCHS),TRAIN_EPOCHS=$(EPOCHS)) python ./src/train.py

# Inference
predict: ## Run single text prediction (usage: make predict TEXT="your text here")
	cd src && python inference.py --text "$(TEXT)"

# Testing
test: ## Run model evaluation on test dataset
	cd src && python inference.py --test --model_path ../checkpoints/best_model.pt --test_file ../data/test_data.parquet

# Container Configuration
CONTAINER_PATH := /oak/stanford/groups/maggiori/GCAP/data/scratch/xwfeng/text_classifier/container/text_classifier.sif
DATA_BIND := /oak/stanford/groups/maggiori/GCAP/data/scratch/xwfeng/text_classifier/data:/app/data
CHECKPOINT_BIND := /oak/stanford/groups/maggiori/GCAP/data/scratch/xwfeng/text_classifier/checkpoints:/app/checkpoints
LOGS_BIND := /oak/stanford/groups/maggiori/GCAP/data/scratch/xwfeng/text_classifier/logs:/app/logs
SRC_BIND := ./src:/app/src
BIND_MOUNTS := $(DATA_BIND),$(CHECKPOINT_BIND),$(LOGS_BIND),$(SRC_BIND)

# Apptainer Commands
apptainer-build: ## Build Apptainer image
	@echo "Building Apptainer container..."
	apptainer build $(CONTAINER_PATH) ./container/text_classifier.def
	@echo "✓ Container built successfully"

apptainer-run: ## Run distributed training in Apptainer container (usage: make apptainer-run [EPOCHS=N])
	@echo "Starting distributed training in container..."
	$(if $(EPOCHS),TRAIN_EPOCHS=$(EPOCHS)) apptainer run --nv --bind $(BIND_MOUNTS) $(CONTAINER_PATH)

apptainer-test: ## Run inference on test dataset in Apptainer container
	@echo "Running inference on test dataset in container..."
	apptainer exec --nv --bind $(BIND_MOUNTS) $(CONTAINER_PATH) /bin/bash -c 'cd /app && . .venv/bin/activate && python src/inference.py --test --model_path checkpoints/best_model.pt --test_file data/test_data.parquet'

apptainer-predict: ## Run single text prediction in container (usage: make apptainer-predict TEXT="your text here")
	@echo "Running prediction in container..."
	apptainer exec --nv --bind $(BIND_MOUNTS) $(CONTAINER_PATH) /bin/bash -c 'cd /app && . .venv/bin/activate && python src/inference.py --text "$(TEXT)"'


gpu-info: ## Show GPU information
	@echo "=== GPU Information ==="
	@nvidia-smi || echo "NVIDIA GPUs not available"
	@echo ""
	@echo "=== PyTorch GPU Detection ==="
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None" 2>/dev/null || echo "PyTorch not available in host environment (use container for PyTorch)"

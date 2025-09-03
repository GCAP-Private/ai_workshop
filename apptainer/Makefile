.PHONY: all build run clean

# Default values - can be overridden via environment variables or .env file
CONTAINER_NAME ?= hello.sif
DEFINITION_FILE ?= container/definition.def
CONTAINER_DIR ?= /oak/stanford/groups/maggiori/GCAP/data/scratch/xwfeng/apptainers
CONTAINER_PATH = $(CONTAINER_DIR)/$(CONTAINER_NAME)

all: build

build:
	@echo "Building Apptainer container..."
	mkdir -p $(CONTAINER_DIR)
	apptainer build $(CONTAINER_PATH) $(DEFINITION_FILE)

run:
	@echo "Submitting job to SLURM..."
	CONTAINER_PATH=$(CONTAINER_PATH) sbatch hello.sbatch

clean:
	@echo "Cleaning up..."
	rm -f $(CONTAINER_PATH)
	rm -f hello.out hello.err
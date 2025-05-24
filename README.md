# SpaceNet 9 Void Project

A machine learning project for satellite imagery (SAR and optical) contrastive learning using the SpaceNet 9 dataset.

## Project Setup

### Environment Variables

This project uses environment variables to manage paths. Create a `.env` file in the project root with the following variables:

```
PROJECT_ROOT=/path/to/your/spacenet9-void
DATA_PATH=/path/to/your/spacenet9-void/data/train/keypoint-crps/train-patch-v2/crops
SAVE_PATH=/path/to/your/spacenet9-void/experiments/exp_01
```

This allows different collaborators to use their own paths without modifying the shared code.

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Data Preparation

The project expects a specific data structure. Run the data preparation script to set up the directory structure:

```bash
python prepare_data.py
```

You'll need to add your SAR and optical image pairs to the crops directory as described in the script output.

## Running the Project

To run the training process:

```bash
cd experiments/exp_01
python run.py
```

## Project Structure

- `src/`: Source code
  - `datafeeder/`: Data loading modules
  - `models/`: Neural network models
  - `training/`: Training loops and utilities
  - `utils/`: Utility functions
- `experiments/`: Experiment configurations and scripts
- `data/`: Dataset directory

## Configuration

The training configuration is defined in `experiments/exp_01/config.json`. You can modify parameters such as batch size, learning rate, and number of epochs there.

## Docker Support

This project includes Docker support for consistent environments across different machines.

### Prerequisites

- Docker and Docker Compose installed on your system
- For GPU support: NVIDIA Container Toolkit (for GPU acceleration)

### Building and Running with Docker

1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Run the project with Docker:
   ```bash
   docker-compose up
   ```

3. For interactive development within the container:
   ```bash
   docker-compose run --rm spacenet9 bash
   ```

### GPU Support

To enable GPU support, uncomment the GPU-related lines in the `docker-compose.yml` file:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Environment Variables

Environment variables are defined in the `docker-compose.yml` file, so you don't need to create a separate `.env` file when using Docker.
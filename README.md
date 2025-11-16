# Shawshank

This is the code for the paper **The Shawshank Redemption of Embodied AI: Understanding and Benchmarking Indirect Environmental Jailbreaks**. For ethical reasons, only a small subset of the dataset is provided. The full dataset will be made available upon paper acceptance through a review process.


## Project Overview

This project is focused on Indirect Environmental Jailbreaks (IEJ), a novel attack vector for embodied AI agents that exploits environmental prompts (e.g., malicious instructions written on walls) to induce unintended behaviors like jailbreaking or denial-of-service (DoS) attacks.

The system includes two primary frameworks:

1. SHAWSHANK: An automatic attack generation framework for indirect environmental jailbreaks.

2. SHAWSHANK-FORGE: An automatic benchmark generation framework for evaluating IEJ attacks.


## Setup and Installation

- Python 3.11 or higherPython 3.11

- The project uses `uv` for building.

You can install the required dependencies by running:

```base
uv python install 3.11
uv sync
uv pip install -r requirements.txt
```
If you want to run the code, use the following command.

```base
uv run main.py
```


## Configuration

The configuration file config.py holds essential API keys and model information for interacting with the Vision-Language Models (VLMs) and other services used in this framework.

1. Update the config.py file with the appropriate API keys.

2. Ensure the paths to models and other resources are correctly set.

Example Configuration

```python
QWEN_VL = {
    "api_key": "YOUR_API_KEY",
    "base_url": "YOUR_API_URL",
    "model": "qwen-vl-max",
}
```

## Usage

### Benchmark Generation
1. The benchmark generation is handled by SHAWSHANK-FORGE, which uses the `collection_scene.py` script to gather images and associated data for generating the benchmark dataset (SHAWSHANK-BENCH).

1. Generate Malicious Instructions: You can generate malicious instructions using the `generate_instructions.py` script. This script takes scene data and image inputs to generate corresponding attack instructions.

### Running the Attack

The `attacktest.py` script is used to simulate attacks using various methods (e.g., MaliciousAttack, RejectAttack, BadRobot, etc.). You can customize the attack method and input data.

To evaluate the performance of the generated attacks, you can use the `evaluate_attack()` function in `attacktest.py`, which takes in attack responses and evaluates them against predefined metrics such as success rates and risk scores.


## Folder Structure

- **config.py**: Configuration file for API keys and model details.
- **generate_instructions.py**: Script for generating attack instructions based on scene data.
- **attack.py**: Contains logic for different attack methods (e.g., `RejectAttack`, `MaliciousAttack`).
- **attacktest.py**: Script to execute and test attacks.
- **collection_scene.py**: Collects and processes scene data for benchmarks.
- **qwen_defense.py**: Implements defensive strategies and evaluations for VLMs.
- **utils.py**: Utility functions for image loading, logging, and other helper tasks.
- **utils_agent.py**: Contains the Agent class for interacting with the models.

# VLM Verifier in Dynamic Environments

## Description

This project evaluates the capability of a Vision-Language Model (VLM) to act as a verifier in dynamic robotic environments. It utilizes the VIMA benchmark (`vima_bench`) to simulate robotic manipulation tasks. 

The core workflow involves:
1.  **Planning:** A VLM inference model generates action plans based on visual input and task prompts.
2.  **Perturbation:** Planned actions are intentionally perturbed before execution to simulate dynamic or imperfect conditions.
3.  **Verification:** A VLM-based verifier assesses the success of the executed (perturbed) action by comparing the environment state before and after.
4.  **Evaluation:** The system tracks task success rates, planner failures, and the accuracy of the verifier under these dynamic conditions.

## Installation
```
pip install -r requirements.txt
```

## Run
```
python run.py task=scene_understanding
```
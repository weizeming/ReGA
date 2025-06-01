# ReGA: Representation-Guided Abstraction \\for Model-based Safeguarding of LLMs

## Usage

1. Download LLMs and datasets from huggingface.co and modify their paths in `language_model.py` and `data.py`
2. Create folder `logs` in current path
3. python main.py --fname EXPERIMENT_NAME --model-name all --abs-model rep --test-loaders all --test-data 1000 --threshold all --pca-dim 8 --state-num 32 --harmful-data 64

You can customize the hyperparameters in the arguments.

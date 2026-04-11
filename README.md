# GL-Mamba for Brain Tumor Segmentation

This project implements a Mamba-based neural network architecture, named GL-Mamba, for medical image segmentation, specifically targeting the BraTS 2021 dataset for brain tumor segmentation.

## Project Structure

- `glmamba/`: Contains the core source code for the project.
  - `data/`: Data loading and preprocessing utilities.
  - `losses/`: Implementation of loss functions.
  - `metrics/`: Evaluation metrics.
  - `models/`: The GL-Mamba model implementation.
  - `utils/`: Helper scripts for argument parsing, checkpoints, etc.
  - `train.py`: The main script for training the model.
  - `eval.py`: The script for evaluating a trained model.
  - `infer.py`: The script for running inference on new data.
- `scripts/`: Contains example SLURM scripts for running training and evaluation on a cluster.
- `requirements.txt`: A list of Python dependencies for this project.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Mamba-LBP
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Dataset

This model is designed to work with the [BraTS 2021 dataset](https://www.med.upenn.edu/cbica/brats2021/).

You will need to organize your data and create a JSON file that specifies the file paths for your training, validation, and testing sets. The training script expects the path to the data root directory and this JSON file.

## Usage

The primary scripts for interacting with the model are `train.py`, `eval.py`, and `infer.py`. You can see all available command-line arguments in `glmamba/utils/argparse.py`.

### Training

Here is an example command to start a training run. You will need to adjust the paths and hyperparameters according to your setup.

```bash
python train.py \
    --data_root_dir /path/to/brats2021/dataset/ \
    --data_list_file_path /path/to/your/datalist.json \
    --log_dir ./logs \
    --batch_size 2 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --d_model 192
```

### Evaluation

To evaluate a trained model, you need to provide the path to the saved model checkpoint.

```bash
python eval.py \
    --data_root_dir /path/to/brats2021/dataset/ \
    --data_list_file_path /path/to/your/datalist.json \
    --checkpoint /path/to/your/model_checkpoint.pth \
    --log_dir ./eval_logs
```

### Inference

To run inference on a single image or a set of images:

```bash
python infer.py \
    --data_root_dir /path/to/brats2021/dataset/ \
    --data_list_file_path /path/to/your/datalist.json \
    --checkpoint /path/to/your/model_checkpoint.pth \
    --log_dir ./infer_logs
```

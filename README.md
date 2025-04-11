# JPEG Quality Prediction

A deep learning model that predicts JPEG compression quality from images. This tool helps you estimate the quality level of JPEG-compressed images.

## Quick Start

1. **Install Dependencies**
```bash
pip install torch torchvision tqdm
```

2. **Download Pre-trained Model**
```bash
# The model.pth file should be in the root directory
```

3. **Make Predictions**
```bash
python inference.py --image_path path/to/your/image.jpg
```

## Detailed Usage

### Making Predictions

The model takes a JPEG image as input and predicts its compression quality level (0-100).

```bash
python inference.py --image_path path/to/image.jpg
```

Example output:
```
Predicted quality: 85.3
```

### Training Your Own Model

1. **Prepare Your Dataset**
   - Create a directory structure:
   ```
   dataset/
   └── train/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```
   - Note: The dataset directory is not tracked in git

2. **Start Training**
```bash
python train.py
```
   - Training progress will be displayed
   - Model checkpoints are saved in `saved_models/` (not tracked in git)
   - Training runs for 1000 epochs by default

### Testing the Model

To evaluate model performance:
```bash
python test.py
```

## Model Details

- Input: RGB images (128x128 pixels)
- Output: Quality score (0-100)
- Architecture: CNN with 4 convolutional layers
- Pre-trained model available: `model.pth`

## Notes

- The following directories/files are not tracked in git:
  - `dataset/` (training data)
  - `saved_models/` (model checkpoints)
  - `__pycache__/` (Python cache)
  - `create_dataset.py` (dataset creation script)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm


## License

[MIT](https://choosealicense.com/licenses/mit/)

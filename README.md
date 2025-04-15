# JPEG Image Restoration and Quality Prediction

A deep learning-based system for JPEG image restoration and quality prediction. This project combines two neural networks to first predict the JPEG compression quality of an image and then restore it to a higher quality version.

## Project Overview

This project addresses the common problem of JPEG compression artifacts in digital images. It consists of two main components:

1. **JPEG Quality Predictor**: A neural network that predicts the compression quality of a JPEG image
2. **Image Restorer**: A U-Net based model that restores JPEG-compressed images to higher quality

The system works by first analyzing the input image to determine its compression quality, then using this information to guide the restoration process.

## Features

- Automatic JPEG quality prediction
- High-quality image restoration
- Tile-based processing for large images
- GPU acceleration support
- Simple command-line interface
- PSNR (Peak Signal-to-Noise Ratio) calculation for quality assessment

## Technical Details

### Models

- **Quality Predictor**: Trained on a dataset of JPEG images with known quality factors
  - Model file: `quality_predictor/saved_models/v6_model_epoch_981_14.5304.pth`
  - Architecture: Custom CNN for quality prediction
- **Restorer**: U-Net architecture with 4 input channels (RGB + quality channel) and 3 output channels (RGB)
  - Model file: `restore_predictor/saved_models/v9_model_epoch_169_0.0005.pth`
  - Architecture: Modified U-Net with quality channel input

### Performance

- The system can process images of any size through tiling
- Average PSNR improvement of X dB (to be filled with actual results)
- Quality prediction accuracy of Y% (to be filled with actual results)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jpeg-restoration.git
cd jpeg-restoration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify model files:
The repository includes two pre-trained model files:
- `quality_predictor/saved_models/v6_model_epoch_981_14.5304.pth`
- `restore_predictor/saved_models/v9_model_epoch_169_0.0005.pth`

Make sure these files are present in their respective directories before running the scripts.

## Usage

### Single Image Processing

To process a single image:

```bash
python single_image_inference.py path/to/your/image.jpg
```

The script will:
1. Load the models
2. Predict the JPEG quality of the input image
3. Restore the image using the predicted quality
4. Save the restored image as `output/restored.png`

### Batch Processing

For batch processing multiple images:

```bash
python merged_inference.py
```

This will process all images in the specified directory and generate restoration results.

## Project Structure

```
jpeg-restoration/
├── quality_predictor/
│   ├── model.py
│   └── saved_models/
│       └── v6_model_epoch_981_14.5304.pth
├── restore_predictor/
│   ├── model.py
│   └── saved_models/
│       └── v9_model_epoch_169_0.0005.pth
├── single_image_inference.py
├── merged_inference.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- Pillow
- numpy

## Future Improvements

- [ ] Add support for different image formats
- [ ] Implement batch processing with progress bar
- [ ] Add web interface for easy usage
- [ ] Support for video processing
- [ ] Real-time processing capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the open-source community for the tools and libraries used in this project
- Special thanks to the researchers whose work inspired this implementation

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/jpeg-restoration](https://github.com/yourusername/jpeg-restoration)
# Image Denoising with Graph Signal Processing

This project implements an image denoising algorithm using graph signal processing techniques. The application utilizes a patch-based approach to reduce noise in images while preserving essential features and details.

## Overview of Graph Signal Processing

Graph signal processing (GSP) is a framework that extends classical signal processing techniques to data defined on irregular domains represented as graphs. In this project, we use GSP to process image patches as signals on a graph structured by a grid. The fundamental idea is to represent the image patches as nodes in a graph, where edges represent the relationships between neighboring pixels. 

### Key Concepts

- **Graph Construction**: The image is divided into smaller overlapping patches, and each patch is represented as a node in a grid graph. 
- **Filtering**: A low-pass filter (specifically, a heat kernel filter) is applied to each patch, which smooths the signal by reducing high-frequency noise while preserving the structure of the underlying image.
- **Parallel Processing**: The algorithm uses parallel processing to enhance performance, allowing multiple patches to be processed simultaneously, improving the overall denoising speed.

The GSP framework allows for more effective denoising compared to traditional methods, especially in preserving edges and fine details in images.

## Installation

To run this application, you'll need to have Python installed along with the following packages:

```bash
pip install streamlit numpy matplotlib pygsp scikit-image
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-denoising.git
    cd image-denoising
    ```

2. Run the application:
    ```bash
    streamlit run app.py
    ```

3. Upload an image and customize the denoising parameters using the sidebar.

## Example

The application provides an interactive interface where users can upload an image, add noise, and then apply the denoising algorithm. The results are displayed side by side for easy comparison of the original, noisy, and denoised images.

### Performance Metrics

After processing, the application calculates and displays the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) scores to evaluate the quality of the denoised image.

## References

For a more in-depth understanding of graph signal processing, you can refer to the following paper:
- **Title**: [Graph Signal Processing: Overview, Applications, and Challenges](https://arxiv.org/abs/1211.0053)  
  **Authors**: Antonio Ortega, Passino, and others

## License

This project is licensed under the MIT License. See the LICENSE file for details.

# SVD-Based RGB Image Compression
 
A Python-based application demonstrating the use of Singular Value Decomposition (SVD) and Linear Algebra for RGB image compression.

# 🖼️ SVD-Based RGB Image Compression

🚀 **[Try the Live App](https://svd-image-compression-3w49daue9ttuvds9del9q3.streamlit.app/)**

A Python application demonstrating Singular Value Decomposition (SVD)
for RGB image compression using low-rank matrix approximation.


## 📌 Project Overview

This project demonstrates how Singular Value Decomposition can be used to obtain a low-rank approximation of an image and reduce the amount of data required to represent it.

A color image consists of three matrices corresponding to the Red, Green and Blue channels.

For each channel:

A = UΣVᵀ

For compression, only the first k singular values are retained:

Aₖ = UₖΣₖVₖᵀ

The three reconstructed RGB channels are then combined to generate the compressed image.

## ✨ Features

- RGB image compression using SVD
- Independent SVD decomposition of Red, Green and Blue channels
- Interactive compression-rank selection
- Original vs reconstructed image comparison
- Compression ratio calculation
- Storage reduction calculation
- Energy retention measurement
- Reconstruction error measurement
- Singular value visualization
- Cumulative energy retention graphs
- Individual RGB channel visualization
- Downloadable compressed image

## 🧮 Mathematical Concepts

The project demonstrates:

- Singular Value Decomposition
- Matrix factorization
- Low-rank approximation
- Eigenvalue-related concepts through singular values
- Frobenius norm
- Matrix approximation
- Data compression
- Linear Algebra applications in Computer Science

## 🛠️ Technologies Used

- Python
- Streamlit
- NumPy
- Pillow
- Matplotlib

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SVD-Image-Compression.git

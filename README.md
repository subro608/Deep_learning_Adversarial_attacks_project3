```markdown
# Deep Learning Project 3: Adversarial Attacks on Image Classifiers

This repository contains implementation of various adversarial attack methods against production-grade image classification models. We investigate how subtle perturbations can significantly degrade the performance of deep neural networks, and evaluate the transferability of these attacks across different model architectures.

## Project Overview

In this project, we develop and analyze several adversarial attack strategies:

1. **FGSM (Fast Gradient Sign Method)** - Basic one-step attack
2. **I-FGSM (Iterative FGSM)** - Multi-step attack with improved effectiveness
3. **PGD (Projected Gradient Descent)** - State-of-the-art attack with random initialization
4. **Patch Attack** - Localized perturbation applied to a small image region

We evaluate these attacks on both the full ImageNet classification space and a restricted subset (classes 401-500), examining how the prediction domain influences attack effectiveness.

## Repository Structure

- `deep_learning_project3.ipynb`: Main Jupyter notebook containing all code and experiments
- `README.md`: This file

## Requirements

The code has been tested with the following dependencies:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- tqdm
- Google Colab (for GPU acceleration)

Install requirements with:
```
pip install torch torchvision numpy matplotlib tqdm
```

## Dataset and Setup

### Obtaining the Dataset
1. Download the TestDataSet.zip from [this Google Drive link](https://drive.google.com/file/d/1u8JqVAnviTdIo8xsjIBTvnSm59K0sZ6l/view?usp=sharing)
2. Upload the zip file to your Google Drive (recommended path: `/MyDrive/Deep_learning_Lp/TestDataSet.zip`)

### Setting Up in Google Colab
Run the following code to mount your Drive and extract the dataset:

```python
import os
import zipfile
from google.colab import drive
drive.mount('/content/drive')
zip_path = '/content/drive/MyDrive/Deep_learning_Lp/TestDataSet.zip'
extract_dir = '/content/TestDataSet'

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extracted files:")
for root, dirs, files in os.walk(extract_dir):
    for file in files:
        print(os.path.join(root, file))
```

The dataset consists of 500 images from 100 classes of ImageNet-1K and needs to be preprocessed using ImageNet normalization:
```python
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])
```

## Attack Visualizations

### FGSM Attack (ε=0.02)
![FGSM Attack Visualization](/images/fgsm.png)

The Fast Gradient Sign Method creates adversarial examples by taking a single step in the direction of the gradient of the loss with respect to the input image. Despite being a one-step method, FGSM creates effective perturbations that remain visually imperceptible to humans while causing significant accuracy drops.

Key characteristics:
- Single-step attack with fast execution (3.60 seconds for 500 images)
- Reduced ResNet-34 top-1 accuracy from 70.4% to 20.2%
- Perturbations are uniform across the entire image but subtle
- Most successful against similar architectures (ResNet50)

### I-FGSM Attack (10 iterations, ε=0.02)
![I-FGSM Attack Visualization](/images/ifgsm.png)

Iterative FGSM applies multiple small steps in the direction of the gradient, recalculating the gradient after each step. This creates more precisely targeted perturbations that completely collapse model performance while remaining visually subtle.

Key characteristics:
- Multi-step attack (24.10 seconds for 500 images)
- Devastating effectiveness: reduced top-1 accuracy to 0.0%
- Perturbations are more structure-aware than FGSM
- Shows moderate transferability across different architectures

### PGD Attack (20 steps, ε=0.02)
![PGD Attack Visualization](/images/pgd.png)

Projected Gradient Descent starts with random initialization and performs constrained optimization with multiple steps. PGD is considered one of the strongest white-box attacks and demonstrates how even within tight perturbation constraints, iterative methods can completely break model performance.

Key characteristics:
- Most computationally intensive (49.13 seconds for 500 images)
- Reduced top-1 accuracy to 0.0% and top-5 to 0.4%
- Random initialization helps find more optimal adversarial examples
- Shows highest transferability to architecturally similar models

### Patch Attack (32×32 patch, ε=0.8)
![Patch Attack Visualization](/images/patch_attack.png)

Unlike full-image perturbations, the patch attack modifies only a small 32×32 region at the center of the image. Despite the localized nature and higher epsilon value, the patch attack is less effective but more stealthy and potentially more practical for real-world applications.

Key characteristics:
- Most time-consuming to generate (143.35 seconds)
- Moderate impact: reduced top-1 accuracy from 70.4% to 62.4%
- Highly visible but localized changes
- Lowest transferability across architectures

## Key Results

### Full ImageNet Evaluation
- **Original ResNet-34**: 70.4% top-1, 93.2% top-5 accuracy
- **FGSM (ε=0.02)**: 20.2% top-1, 36.8% top-5 accuracy
- **I-FGSM (ε=0.02)**: 0.0% top-1, 1.2% top-5 accuracy
- **PGD (ε=0.02)**: 0.0% top-1, 0.4% top-5 accuracy
- **Patch Attack (ε=0.8)**: 62.4% top-1, 87.6% top-5 accuracy

### Restricted Classes (401-500)
- **Original ResNet-34**: 87.6% top-1, 99.2% top-5 accuracy
- **FGSM (ε=0.02)**: 35.2% top-1, 60.6% top-5 accuracy
- **I-FGSM (ε=0.02)**: 0.8% top-1, 7.2% top-5 accuracy
- **PGD (ε=0.02)**: 0.2% top-1, 5.2% top-5 accuracy
- **Patch Attack (ε=0.8)**: 82.4% top-1, 98.0% top-5 accuracy

### Transferability Analysis
We tested our adversarial examples against DenseNet121, ResNet50, and MobileNetV3, finding:
- ResNet50 showed highest vulnerability to transferred attacks (30-45% transferability)
- MobileNetV3 demonstrated superior robustness (15-25% transferability)
- Full-image attacks transferred better than patch attacks
- Architectural similarity correlated with higher transferability rates

## Comparison Visualizations

### Accuracy Drop Comparison
![Accuracy Drop Comparison](/images/accuracy_drop.png)

This visualization compares the top-1 accuracy drop (%) across different models and attack types. The chart highlights how transformer-based architectures like ViT-B/16 remain significantly more robust against gradient-based attacks compared to traditional CNN architectures.

### Transferability Heatmap
![Transferability Heatmap](/images/transferability_heatmap.png)

The transferability heatmap visualizes how effectively adversarial examples generated for ResNet-34 transfer to other model architectures. Darker colors indicate higher transferability rates, showing clear architectural patterns in vulnerability.

## Usage

1. Open the notebook in Google Colab
2. Mount your Google Drive and extract the dataset as shown above
3. Run the notebook cells sequentially to reproduce our results

```python
# Example code for running basic FGSM attack
epsilon = 0.02
fgsm_images = create_adversarial_examples_fgsm(model, images, labels, epsilon)
```

## Team Members
- Subhrajit Dey
- Yogarajalakshmi Sathyanarayanan
- Pooja Thakur

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.
- Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). Adversarial patch. arXiv preprint arXiv:1712.09665.
```

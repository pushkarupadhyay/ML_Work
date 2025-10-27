# Setup Guide

This guide will help you set up your environment for working with the ML_Work repository.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- (Optional) CUDA-enabled GPU for deep learning tasks

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pushkarupadhyay/ML_Work.git
cd ML_Work
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to avoid dependency conflicts.

**Using venv (built-in):**
```bash
python -m venv venv
```

**Activate the virtual environment:**

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Some packages like TensorFlow and PyTorch are large. The installation may take several minutes.

### 4. Install Jupyter Kernel (Optional)

If you plan to use Jupyter notebooks:

```bash
python -m ipykernel install --user --name=ml_work --display-name="Python (ML_Work)"
```

### 5. Verify Installation

```bash
python -c "import numpy, pandas, sklearn, tensorflow, torch; print('All packages imported successfully!')"
```

## GPU Support

### TensorFlow GPU

If you have an NVIDIA GPU and want to use it with TensorFlow:

```bash
pip install tensorflow-gpu
```

Make sure you have the appropriate CUDA and cuDNN versions installed.

### PyTorch GPU

For PyTorch with GPU support, visit [pytorch.org](https://pytorch.org/) and follow the installation instructions for your CUDA version.

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Jupyter
- Pylance
- Python Docstring Generator

### PyCharm

1. Open the project folder
2. Set the Python interpreter to your virtual environment
3. Enable Jupyter notebook support

## Common Issues

### Import Errors

If you encounter import errors, make sure:
1. Your virtual environment is activated
2. All dependencies are installed
3. You're running Python from the project root directory

### Memory Issues

When working with large datasets or models:
1. Close unnecessary applications
2. Use data generators instead of loading entire datasets into memory
3. Consider using cloud computing resources (Google Colab, AWS, etc.)

## Next Steps

After setup, you can:
1. Explore the `notebooks/` directory for examples
2. Check out the `projects/example_project/` for a template structure
3. Start your own ML project in the `projects/` directory

## Getting Help

If you encounter issues:
1. Check the documentation in the `docs/` directory
2. Review the README files in each directory
3. Open an issue on GitHub

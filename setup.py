from setuptools import setup, find_packages

setup(
    name="task_aware_sam_lora",
    version="0.1.0",
    description=(
        "A hypernetwork that generates task‑specific LoRA adapters "
        "for SAM’s mask decoder based on textual prompts"
    ),
    author="Yash Thube",
    author_email="thubeyash09@gmail.com",
    url="https://github.com/thubZ09/task-aware-sam-lora",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        #segment-anything must be installed via git:
        # pip install git+https://github.com/facebookresearch/segment-anything.git
        "opencv-python>=4.5.0",
        "albumentations>=1.3.0",
        "transformers>=4.25.0",
        "sentence-transformers>=2.2.0",
        "pycocotools>=2.0.6",
        "datasets>=2.8.0",
        "pandas>=1.5.0",
        "wandb>=0.13.0",
        "tqdm>=4.64.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.11.0",
        "plotly>=5.11.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.5.0",
            "rich>=12.0.0",
            "loguru>=0.6.0",
            "einops>=0.6.0",
            "timm>=0.6.0",
            "scikit-learn>=1.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tasl-train=train:main",
            "tasl-demo=demo:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

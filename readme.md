
# **Text-to-Image Generation Using GANs on the Birds Dataset**


## **Project Overview**  
This project implements a **Generative Adversarial Network (GAN)** to generate realistic bird images conditioned on textual descriptions. The GAN architecture uses text embeddings as inputs to the Generator and ensures that the generated images match the provided text descriptions.  

The **CUB-200-2011 Birds Dataset** is used, which contains bird images and associated textual descriptions. The project is implemented in **PyTorch** and optimized for the **MPS backend** on Apple Silicon.

---

## **Project Objectives**  
1. Implement a **conditional GAN** to generate images based on textual descriptions.  
2. Use the **CUB-200-2011 dataset** as the source for training.  
3. Address challenges such as training instability and hardware limitations.  
4. Set up a modular pipeline for training, inference, and evaluation.

---

## **Dataset**  
- **Dataset Used**: [CUB-200-2011 Birds Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  
- **Format**:  
   - Images resized to **64x64 pixels**.  
   - Text descriptions converted into **precomputed embeddings**.  
   - Stored in an **HDF5** file for efficient data loading.  
- **Splits**: Train, Validation, and Test.  

---

## **Project Structure**  

```plaintext
├── config.yaml
├── dataset
│   └── birds.hdf5
├── model
│   ├── __pycache__
│   │   ├── gan_cls.cpython-310.pyc
│   │   └── gan_factory.cpython-310.pyc
│   ├── gan_cls.py
│   └── gan_factory.py
├── notebook.ipynb
├── pytorch files
│   ├── birds
│   │   ├── disc_190.pth
│   │   └── gen_190.pth
│   └── birds_cls
│       ├── disc_190.pth
│       └── gen_190.pth
├── readme.md
├── requriments.txt
├── src
│   ├── __pycache__
│   │   ├── trainer.cpython-310.pyc
│   │   ├── txt2image_dataset.cpython-310.pyc
│   │   └── utils.cpython-310.pyc
│   ├── demo_text.py
│   ├── runtime.py
│   ├── trainer.py
│   ├── trainer_demo.py
│   ├── txt2image_dataset.py
│   └── utils.py
└── tree.log

9 directories, 23 files
```

---

## **Requirements**  

Before running the project, ensure the following dependencies are installed:

### **Dependencies**  
Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

**Key Libraries**:  
- PyTorch (with MPS backend for Apple Silicon)  
- h5py  
- PIL (Pillow)  
- NumPy  
- PyYAML  

---

## **Setup and Usage**  

### **1. Clone the Repository**  
Clone the project repository to your local machine:

```bash
git clone <repository_url>
cd project_root
```

---

### **2. Configure the Project**  
Modify the **config.yaml** file to set dataset paths, save directories, and hyperparameters.  

Example `config.yaml` content:

```yaml
birds_dataset_path: "/path/to/birds_dataset.hdf5"
flowers_dataset_path: "/path/to/flowers_dataset.hdf5"
```

---

### **3. Run the Training**  
To train the model, execute the `runtime.py` script:

```bash
python runtime.py
```

**Training Parameters**:  
- Model type: GAN  
- Batch size: 64  
- Epochs: 200  
- Device: MPS (for MacBook M1/M2)  

---

### **4. Run Inference**  
If you already have trained Generator and Discriminator models, set the `inference` flag in `runtime.py` to `True` and provide checkpoint paths:

```python
args = easydict.EasyDict({
    'inference': True,  
    'pre_trained_disc': 'checkpoints/disc_100.pth',  
    'pre_trained_gen': 'checkpoints/gen_100.pth',  
})
```

Run the script:

```bash
python runtime.py
```

The generated images will be saved in the **`results/`** directory.

---

## **Challenges Faced**  
1. **Hardware Limitations**: Training on the **MPS backend** was slower compared to CUDA.  
2. **Training Instability**: Loss oscillations during GAN training were observed.  
3. **Low Resolution**: Generated images are limited to **64x64 pixels** due to resource constraints.  

---

## **Future Improvements**  
1. Train for more epochs to improve model convergence.  
2. Implement advanced GAN architectures like **StackGAN** or **AttnGAN** for higher resolution images.  
3. Use pre-trained models like **CLIP** or **BERT** for better text embeddings.  
4. Add evaluation metrics such as **Inception Score** and **FID** for quantitative analysis.  
5. Train on cloud-based GPUs (Google Colab, AWS) for faster results.

---

## **References**  
1. Ian Goodfellow et al., *Generative Adversarial Networks*, NIPS, 2014.  
2. Han Zhang et al., *StackGAN: Text to Photo-realistic Image Synthesis*, ICCV, 2017.  
3. Tao Xu et al., *AttnGAN: Fine-Grained Text to Image Generation*, CVPR, 2018.  
4. CUB-200-2011 Dataset: [CUB Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).  
5. PyTorch Documentation: [https://pytorch.org](https://pytorch.org).  

---

## **Contact**  
For questions or suggestions, feel free to reach out to:  
- **Name**: Kshitij Dhannoda 
- **Email**: dkshitij@umich.edu  
 
---
## Youtube Links:
   - [Link to the channel](https://www.youtube.com/channel/UCAMULu7AVoDz36yfiQFSocw)

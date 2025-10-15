# 🧠 Image Classification with Hugging Face using Vision Transformers (ViT)

This project demonstrates the power and versatility of **Visual Transformers (ViT)** for multiple image understanding tasks using the **Hugging Face Transformers** library.  
You’ll explore **simple classification**, **zero-shot image classification**, and **zero-shot object detection** — all powered by state-of-the-art pretrained transformer models.

---

## 🚀 Project Overview

The goal of this project is to understand how **transformer-based vision models** can be applied to various computer vision challenges without extensive retraining.

### 🖼️ Tasks Covered
1. **Simple Image Classification**  
   - Classifies an image into predefined categories using a pretrained `vit-base-patch16-224` model.  

2. **Zero-Shot Image Classification**  
   - Uses the `openai/clip-vit-large-patch14` model to classify images into categories **never seen during training**.  

3. **Zero-Shot Object Detection**  
   - Detects and localizes multiple objects in an image using `google/owlvit-base-patch32`, even if the model hasn’t been explicitly trained for those categories.  

---

## 🧩 Tech Stack
- **Programming Language:** Python  
- **Frameworks & Libraries:**  
  - 🧠 [Transformers](https://huggingface.co/docs/transformers) (Hugging Face)  
  - 🔥 PyTorch  
  - 🖼️ PIL (Pillow)  
  - 📊 Matplotlib  
  - 🌐 Requests  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Harsha2001-creater/Image-Classification-with-Hugging-Face-using-Vision-Transformers-ViT-.git
cd Image-Classification-with-Hugging-Face-using-Vision-Transformers-ViT-
pip install torch torchvision transformers matplotlib pillow requests
```
# 🧠 Project Workflow

## 🧩 Step-by-Step Process

### 1. Image Classification
- Load a pretrained **Vision Transformer (ViT)** model from Hugging Face.  
- Import and configure the **feature extractor** to preprocess images.  
- Load and visualize an input image.  
- Convert the image into tensor format suitable for ViT.  
- Pass the processed image through the model to generate predictions.  
- Identify and display the most probable class label for the image.  

---

### 2. Zero-Shot Image Classification
- Load the **CLIP (Contrastive Language–Image Pretraining)** model from OpenAI via Hugging Face.  
- Prepare a list of **candidate labels** (e.g., car, ship, road, fruit, etc.).  
- Input both the image and candidate labels into the **CLIP processor**.  
- Perform inference to calculate similarity scores between the image and each label.  
- Rank the labels by confidence and display the top prediction.  

---

### 3. Zero-Shot Object Detection
- Load the **OwlViT (Object-World Learning Vision Transformer)** model.  
- Define a set of **textual object queries** (e.g., book, hat, camera).  
- Provide both the image and text queries to the processor.  
- Perform inference to detect and localize objects within the image.  
- Retrieve and post-process **bounding box coordinates** and **confidence scores**.  
- Draw bounding boxes and labels on the image to visualize detected objects.  

---

## 🎯 Key Learnings
- How to **load and fine-tune transformer models** for vision tasks.  
- The difference between **simple classification**, **zero-shot classification**, and **object detection**.  
- How to **visualize predictions** with bounding boxes using PIL.  
- The **power of transformer-based models** to generalize to unseen data.  

---

## 📈 Results Snapshot

| Task | Model Used | Example Output |
|------|-------------|----------------|
| Image Classification | ViT Base Patch 16 | Predicted: `Loafer` |
| Zero-Shot Classification | CLIP ViT Large Patch 14 | Top Label: `Car (98.7%)` |
| Zero-Shot Object Detection | OwlViT Base Patch 32 | Detected: `Book`, `Hat`, `Camera` |

---

## 🏁 Summary
Through this project, you explored how **Visual Transformers** revolutionize image understanding tasks by enabling both **classification** and **object detection** — even in **zero-shot** scenarios.  
This hands-on exercise showcases how Hugging Face models can be directly applied to practical computer vision challenges with minimal setup.

---

## 📚 References
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [ViT: An Image is Worth 16x16 Words (Google Research)](https://arxiv.org/abs/2010.11929)
- [CLIP: Connecting Text and Images (OpenAI)](https://arxiv.org/abs/2103.00020)
- [OwlViT: Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)



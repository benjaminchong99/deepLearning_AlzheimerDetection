# Alzheimer's Detection with CNNs
Alzheimer's disease (AD) is a progressive neurodegenerative condition leading to memory loss, particularly affecting episodic memory functions such as context recall. It stands as the leading cause of dementia, significantly impacting memory, thinking, language, judgment and behavior. In addition, Alzheimer's disease is irreversible and progressive. Alzheimer's is evaluated by identifying certain symptoms and ruling out other possible causes of dementia, often utilizing medical exams like CT, MRI or PET scans of the head. While there is no cure, medications can help to slow the disease's progression and manage the symptoms.

The project aims to evaluate pre existing computer vision models, in particular CNN and Inception v3 for Alzheimer's detection. It seeks to determine the most effective model for prediction accuracy.

## Getting Started
Clone this repository
```
git clone https://github.com/benjaminchong99/deepLearning_AlzheimerDetection.git
```

## Requirements
1. Install the dependencies:
`pip install -r requirements.txt`
2. Ensure ad_labels.py, datasetv2.py, datasetv3.py and dataset.zip are all in your environment.
3. Unzip dataset.zip
4. Run inceptionv3_smote_3.ipynb for a full breakdown of inceptionv3 models and standard 2-layer CNN
5. Run DL_proj for learning rate experiments.

## Project Structure
```
deepLearning_AlzheimerDetection
├── DL_proj.ipynb
├── Dataset 
├── README.md
├── ad_labels.py
├── archive.zip
├── dataset.py (Dataset initialisation class)
├── datasetv2.py (Dataset init class with Weighted Random Sampler)
├── inception_models (Trained Inception Models)
├── inceptionv3.ipynb (Inception v3 base model training)
├── inceptionv3_smote.ipynb (Inception v3 base model with SMOTE)
├── requirements.txt
└── saved_models (Trained custom models)
```

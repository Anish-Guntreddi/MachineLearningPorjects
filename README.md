# ML Engineering Portfolio

A collection of 12 end-to-end machine learning projects spanning computer vision, NLP, speech, recommenders, time series, anomaly detection, and multimodal fusion. Each project follows the same structure — problem definition, dataset acquisition, model implementation, training, and evaluation — and includes a runnable notebook, standalone training/inference scripts, a model card, and recorded results. A Streamlit app (`portfolio_app/`) ties the projects together into a single browsable demo.

## Projects

| # | Project | Description | Key tech | Best model | Headline result |
|---|---------|-------------|----------|------------|------------------|
| 01 | [Image Classification](01_Image_Classification) | CIFAR-10 / Tiny ImageNet classification | PyTorch, torchvision, timm | resnet18 | 93.2% test accuracy |
| 02 | [Object Detection](02_Object_Detection) | Bounding-box detection on COCO / Pascal VOC | PyTorch, torchvision, pycocotools | fasterrcnn_resnet50_fpn | 78.3 mAP@50 |
| 03 | [Instance Segmentation](03_Instance_Segmentation) | Pixel-level instance masks on COCO / Cityscapes | PyTorch, torchvision, pycocotools | mask_rcnn | 37.2 mask mAP |
| 04 | [Text Classification](04_Text_Classification) | Sentiment/topic classification on IMDb / AG News | Hugging Face Transformers, datasets | bert | 92.5% test accuracy |
| 05 | [Text Generation](05_Text_Generation) | Language modeling / fine-tuning on WikiText | Hugging Face Transformers | gpt2 | 29.4 test perplexity |
| 06 | [Machine Translation](06_Machine_Translation) | Seq2seq translation on WMT / Europarl | PyTorch Transformer, sacrebleu | transformer | 34.2 BLEU |
| 07 | [Speech Emotion Recognition](07_Speech_Emotion_Recognition) | Emotion classification from speech audio (RAVDESS / CREMA-D) | librosa, PyTorch | cnn_lstm | 78.5% test accuracy |
| 08 | [Automatic Speech Recognition](08_Automatic_Speech_Recognition) | Speech-to-text on LibriSpeech / Common Voice | Hugging Face Transformers, jiwer | whisper_tiny | 7.6% WER |
| 09 | [Recommender System](09_Recommender_System) | Rating prediction / ranking on MovieLens | PyTorch, surprise | ncf (neural collaborative filtering) | 0.92 RMSE, 0.58 NDCG@10 |
| 10 | [Time Series Forecasting](10_Time_Series_Forecasting) | Forecasting on M4 / ETT datasets | PyTorch, statsmodels | lstm | 8.7% MAPE |
| 11 | [Anomaly Detection](11_Anomaly_Detection) | Outlier/intrusion detection (KDD Cup 99 and industrial IoT data) | PyTorch (VAE), scikit-learn | vae | 0.94 AUC-ROC |
| 12 | [Multimodal Fusion](12_Multimodal_Fusion) | Combining vision, audio, and text (MELD) | PyTorch, attention fusion | attention_fusion | 85.3% test accuracy |

Metrics above are read directly from each project's `results.yaml` (recorded training runs); see each project's README and `model_card.yaml` for full methodology, dataset details, and caveats.

## How this repo is organized

- `01_..._12_.../` — one directory per project, each self-contained with:
  - `README.md` — problem definition, dataset, and approach
  - `data_loader.py`, `models.py`, `train.py`, `inference.py`, `utils.py` — implementation
  - `notebook.ipynb` / `notebook.py` — the same workflow as a runnable notebook
  - `model_card.yaml`, `results.yaml` — model documentation and recorded metrics
- `notebook_templates/` and `notebook_configs/` — shared cell templates (per domain: vision, NLP, audio, tabular, multimodal) and per-project YAML configs used by `generate_notebooks.py` to generate each project's notebook
- `generate_notebooks.py` / `generate_precomputed.py` — scripts that generate the notebooks and precompute/serialize results
- `portfolio_app/` — a Streamlit app that presents the projects as a browsable portfolio site
- `requirements.txt` — shared Python dependencies (PyTorch, Transformers, librosa, scikit-learn, etc.) for running the projects

## Getting started

```bash
pip install -r requirements.txt
```

Each project can be run independently via its `train.py` / `notebook.ipynb`, or explored together via the Streamlit app:

```bash
streamlit run portfolio_app/app.py
```

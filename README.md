# 📢 DeepHateDetect: Explainable Bengali Abusive comments classification Using Transformers and LLM


DeepHateDetect is a comprehensive research project that benchmarks 13 models—from traditional machine learning algorithms to large language models (LLMs)—for detecting abusive and hateful Bengali comments. This work integrates explainable AI techniques (LIME) to provide transparent, word-level justifications for each model's prediction.

## 🔍 Key Contributions

✅ Comparative study of 13 models across 4 families:

* Traditional ML: MNB, LR, SVM, RF, KNN
* Sequential models: LSTM, BiLSTM, CNN-LSTM, GRU
* Transformers: BERT, BanglaBERT, RoBERTa
* Large Language Models: LLaMa 3-8B (4-bit quantized)

✅ Achieved 99% accuracy using LLaMa 3-8B on 2.5K samples.
✅ Used LIME to interpret model decisions and highlight toxic language cues.
✅ Built a reproducible pipeline for Bengali hate speech detection.

---

## 🗂 Dataset

* Size: 2,577 human-annotated Bengali abusive comments
* Source: Mixed collection from social platforms
* Labels: Multi-class (Abusive, Hate)

---

## 🧰 Methodology 

<img src="https://github.com/Diagrams ENT (1).png" alt="After Pre-processing">
<sub>Figure 1: Workflow Overview for End-to-End Text Classification</sub>
### 1. Preprocessing

* Tokenization (BERT tokenizer for transformer models)
* Label encoding and stratified train-test split

![BERT Tokenizer](./images/bert_tokenizer.png) <sub>Figure 2: BERT Tokenization Process</sub>

### 2. Model Training

* Implemented using PyTorch, HuggingFace Transformers, and Scikit-learn
* All models evaluated under consistent experimental settings
* LLaMa 3-8B run using 4-bit quantization (QLoRA) for memory efficiency

### 3. Explainability

* LIME used to explain prediction logic
* Highlights abusive keywords that triggered model prediction

![Explainable AI Results](./images/explainable_insights.png) <sub>Figure 3: Insights into Model Predictions for Bengali Corpus Using Explainable AI</sub>

---

## 📊 Results Summary

| Model              | Accuracy |
| ------------------ | -------- |
| Naive Bayes        | \~83%    |
| SVM                | \~86%    |
| LSTM               | \~90%    |
| BanglaBERT         | \~94%    |
| RoBERTa-base       | \~95%    |
| LLaMa 3-8B (4-bit) | ✅ 99%    |

✅ LLaMa 3-8B clearly outperforms all other models on this low-resource, compact dataset.

---

## 🔎 Explainability Example

> Input: "এই ছেলেটা সবসময় বাজে কথা বলে। সমাজের জন্য ক্ষতিকর।"
> LIME Highlights: "বাজে", "ক্ষতিকর"
> Prediction: Hate Speech
> Confidence: 98%

---

## 💻 Installation & Usage

Clone the repo:

```bash
git clone https://github.com/your-username/DeepHateDetect.git
cd DeepHateDetect
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference or training:

```bash
python run_model.py --model llama --quantize 4bit
```

Generate explanations:

```bash
python explain.py --input "আপনি একজন অসভ্য লোক।"
```

---

## 📁 Project Structure

```
DeepHateDetect/
│
├── data/                  # Bengali dataset (cleaned & labeled)
├── models/                # All model implementations
├── explainability/        # LIME visualizations
├── images/                # Figures used in README
├── results/               # Evaluation metrics
├── run_model.py           # Unified script for training/inference
├── explain.py             # Run LIME explanations
└── requirements.txt
```

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{deephatedetect2025,
  title={DeepHateDetect: A Comparative Study of Traditional, Sequential, Transformer, and Large Language Models for Hate Speech Detection in Bengali using Explainable AI},
  author={Your Name(s)},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-username/DeepHateDetect}}
}
```

---

## 🙌 Acknowledgements

* HuggingFace Transformers
* LLaMa 3 by Meta
* Scikit-learn, PyTorch, LIME
* Bengali NLP Community for dataset inspiration

---

Let me know when you’re ready to plug in image filenames and repo links—I can adjust those paths!

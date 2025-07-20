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



## 🗂 Dataset

* Size: 2,577 human-annotated Bengali abusive comments
* Source: Mixed collection from social platforms
* Labels: Multi-class (Abusive, Hate)



## 🧰 Methodology 

<img width="3510" height="2417" alt="Diagrams ENT (1)" src="https://github.com/user-attachments/assets/a3243dda-1e9c-4184-a1fb-77d2894dbd1e" />
<sub>Figure 1: Workflow Overview for End-to-End Text Classification</sub>
### 1. Preprocessing

* Tokenization (BERT tokenizer for transformer models)
* Label encoding and stratified train-test split

<img width="1311" height="1164" alt="BERTD (1)" src="https://github.com/user-attachments/assets/c560151b-68ea-40c2-b1da-ee0691affae5" />
 <sub>Figure 2: BERT Tokenization Process</sub>

### 2. Model Training

* Implemented using PyTorch, HuggingFace Transformers, and Scikit-learn
* All models evaluated under consistent experimental settings
* LLaMa 3-8B run using 4-bit quantization (QLoRA) for memory efficiency

### 3. Explainability

* LIME used to explain prediction logic
* Highlights abusive keywords that triggered model prediction

<img width="1058" height="222" alt="Hate_AI (1)" src="https://github.com/user-attachments/assets/111e9a3c-4dda-47cb-8ec1-ff757874f4cb" />

 <sub>Figure 3: Insights into Model Predictions for Bengali Corpus Using Explainable AI</sub>

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





## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{deephatedetect2025,
  title={DeepHateDetect: A Comparative Study of Traditional, Sequential, Transformer, and Large Language Models for Hate Speech Detection in Bengali using Explainable AI},
  author={Abdullah Al Shafi},
  year={2025},
  publisher={NCIM 2025},
  howpublished={\url{https://github.com/your-username/DeepHateDetect}}
}
```




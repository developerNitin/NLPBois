# Hate Speech Detection: Model Benchmarking & Analysis

This project explores various Natural Language Processing (NLP) architectures to determine the most effective method for detecting hate speech. The study compares traditional baseline models, dense neural networks, and state-of-the-art Transformer architectures.

## 🚀 Executive Summary
After rigorous evaluation, **RoBERTa-Base** was identified as the superior model. It provides the best balance between overall accuracy and the critical ability to recall minority-class instances (hate speech) that traditional models often miss.

---

## 📊 Performance Leaderboard

The models are ranked based on their **Weighted F1-Score**, which accounts for class distribution, ensuring that performance on the majority class doesn't mask failures in the minority class.

| Rank | Model | Weighted F1-Score |
| :--- | :--- | :--- |
| **1** | **RoBERTa-Base** | **0.8739** |
| 2 | BERT-Base | 0.8597 |
| 3 | FFNN (FastText) | 0.8498 |
| 4 | MNB (TF-IDF) | 0.7689 |

---

## 🔍 Detailed Model Analysis

### 🏆 The Top Performer: RoBERTa-Base
RoBERTa-Base emerged as the optimal solution. 
* **Detection Power:** It achieved the highest hate speech recall (**0.4755**), which is vital for moderation tasks where missing a harmful comment is more costly than a false positive.
* **Contextual Depth:** Its robust pre-training enables it to understand nuanced social contexts and linguistic patterns that define hate speech.

### 🥈 The Runner Up: BERT-Base
While competitive with a Weighted F1 of **0.8597**, BERT-Base adopted a more "conservative" stance. This resulted in a significantly lower hate speech recall (**0.3636**), meaning it missed more instances of harmful content compared to RoBERTa.

### ⚡ The Efficient Alternative: FFNN (FastText)
The Feed-Forward Neural Network performed surprisingly well, nearly matching RoBERTa in manual detection tests. With a Weighted F1 of **0.8498**, it serves as an excellent computationally efficient alternative for environments where Transformer latency is a concern.

### 📉 The Baseline: Multinomial Naive Bayes (MNB)
Trailing significantly (F1: **0.7689**), MNB highlighted the limitations of bag-of-words (TF-IDF) approaches. It struggled with a critically low recall (**0.2517**), failing to capture the contextual complexity of modern hate speech.

---

## ⚖️ Impact of Class Imbalance

The data reveals a significant gap between **Weighted** and **Macro** recall metrics. This "Performance Delta" illustrates how much the model relies on the majority class for its accuracy score.

| Model | Weighted vs. Macro Recall Delta |
| :--- | :--- |
| **MNB (TF-IDF)** | +0.1393 |
| **BERT-Base** | +0.1230 |
| **RoBERTa-Base** | +0.0963 |
| **FFNN (FastText)** | +0.0956 |

> **Finding:** RoBERTa and FFNN show a smaller delta, suggesting they are more resilient to class imbalance and provide more consistent predictions across both "Hate" and "Non-Hate" categories.

---

## 🛠️ Technical Stack
* **Transformers:** `RoBERTa`, `BERT` (via HuggingFace)
* **Neural Networks:** Feed-Forward (FFNN) with `FastText` Embeddings
* **Baselines:** Multinomial Naive Bayes with `TF-IDF`
* **Tools:** Python, Scikit-Learn, PyTorch, Jupyter Notebooks

## 📂 Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

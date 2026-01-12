# DeepSurvey: AI-Powered Academic Domain Explorer
# DeepSurvey: AI 驱动的学术领域探索与可视化平台

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DeepSurvey** is a next-generation academic research tool designed to visualize knowledge landscapes. Unlike traditional keyword searches that return linear lists, DeepSurvey uses **state-of-the-art embedding models** and **topological clustering algorithms** to map research fields into interactive point clouds, automatically identifying sub-topics and trends using Large Language Models (LLMs).

**DeepSurvey** 是下一代学术调研工具，旨在通过可视化技术重塑知识发现过程。不同于传统线性列表式的搜索结果，DeepSurvey 利用**最先进的嵌入模型**和**拓扑聚类算法**，将枯燥的论文列表转化为交互式点云图，并利用大语言模型（LLM）自动识别和总结细分研究主题。

---

## ✨ Key Features (核心特性)

### 🧠 1. Dual-Stage Dimensionality Reduction (双重降维架构)
We employ a mathematically rigorous approach to balance clustering accuracy with visual clarity:
* **High-Dimensional Clustering (10D):** Data is first reduced to a 10-dimensional manifold using UMAP to preserve complex topological structures for the clustering algorithm.
* **Low-Dimensional Visualization (2D):** A separate projection is generated specifically for the UI, ensuring the visual map is aesthetically pleasing without compromising the underlying clustering logic.

**双重降维策略**：我们采用了严谨的数学方法来平衡聚类准确性与可视化清晰度。首先利用 UMAP 将数据降维至 10 维流形以供算法捕捉复杂的拓扑结构，随后单独生成 2 维投影用于前端展示。这解决了“为了画图而牺牲聚类精度”的常见问题。

### 🧩 2. Robust Clustering with HDBSCAN (基于 HDBSCAN 的鲁棒聚类)
Moving beyond K-Means, we use **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
* **Automatic Cluster Detection:** No need to specify the number of topics ($K$) in advance.
* **Noise Handling:** Automatically identifies and isolates outlier papers (noise), ensuring that the generated topics are coherent and high-quality.

**HDBSCAN 智能聚类**：摒弃了传统的 K-Means，我们采用基于密度的层次聚类算法。它不需要预先指定聚类数量，并且能自动识别并过滤“噪音”（离群点），确保生成的每个主题都具有高度的一致性。

### 🔌 3. Hybrid Embedding Engine (混合嵌入引擎)
DeepSurvey offers a flexible strategy for vectorization:
* **S2 Mode (Specter):** Leverages Semantic Scholar's pre-computed embeddings for maximum speed and quality.
* **Local Mode (On-Device AI):** A fallback mechanism that runs `all-MiniLM-L6-v2` locally on your CPU/GPU, ensuring 100% data coverage even for obscure or new papers.

**混合嵌入引擎**：提供灵活的向量化策略。用户可以在追求极致速度的 **S2 模式**（使用 Semantic Scholar 预训练向量）和追求全量覆盖的 **本地模式**（本地运行轻量级模型）之间自由切换。

### 🤖 4. LLM-Powered Insight Generation (LLM 智能洞察)
Integrated with **LiteLLM**, the system supports OpenAI, Anthropic, Gemini, DeepSeek, and any OpenAI-compatible API. It analyzes the representative papers in each cluster to generate **specific, technical topic labels** (e.g., "Quantum Error Correction" instead of generic "Quantum Computing").

**LLM 智能总结**：通过集成 LiteLLM，支持接入所有主流大模型。系统会自动分析每个聚类中的核心论文，生成**具体的、技术性的主题标签**，彻底告别笼统的分类命名。

---

## 🛠️ Architecture (技术架构)

* **Frontend:** Streamlit + Streamlit Extras (Modern UI components)
* **Data Source:** Semantic Scholar API (Graph API)
* **NLP & Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Math & ML:** UMAP (Manifold Learning), HDBSCAN (Clustering), NumPy, Pandas
* **Visualization:** Plotly Express (Interactive Point Clouds)
* **LLM Orchestration:** LiteLLM

---

## 🚀 Getting Started (快速开始)

### Prerequisites (前置要求)
* Python 3.9 or higher
* (Optional) Semantic Scholar API Key
* (Optional) OpenAI / DeepSeek / Claude API Key

### Installation (安装)

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/DeepSurvey.git](https://github.com/yourusername/DeepSurvey.git)
    cd DeepSurvey
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    streamlit run app.py
    ```

---

## 📸 Screenshots (界面预览)

![](assets/demo.png)

> **Interactive Filtering:** Filter the knowledge graph by citations, publication year, or specific topics in real-time.
>
> **交互式筛选**：实时根据引用量、发表年份或特定主题筛选知识图谱。

---

## ⚙️ Configuration (配置说明)

| Setting | Description |
| :--- | :--- |
| **S2 API Key** | Optional. Without it, the app may be rate-limited by Semantic Scholar. |
| **Embedding Source** | Select **S2** for speed/quality (might drop data) or **Local** for completeness. |
| **LLM Provider** | Choose between OpenAI, Anthropic, Gemini, or OpenAI-Compatible (e.g., LocalAI/vLLM). |

---

## 🤝 Contribution (贡献)

Contributions are welcome! Please feel free to submit a Pull Request.

欢迎各种形式的贡献！无论是新功能建议还是 Bug 修复，请随时提交 Pull Request。

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
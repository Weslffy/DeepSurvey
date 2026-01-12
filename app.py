import streamlit as st
import pandas as pd
import warnings

# 屏蔽非关键警告
warnings.filterwarnings("ignore")

# --- UI Extras ---
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.badges import badge

# --- Core Modules ---
from core.fetcher import PaperFetcher
from core.processor import DataProcessor
from core.llm_engine import LLMSummarizer
from utils.visuals import plot_paper_map

# 1. 页面基础配置
st.set_page_config(
    page_title="DeepSurvey",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# 2. 注入自定义 CSS (极简主义设计)
st.markdown("""
<style>
    /* 全局字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 隐藏 Streamlit 默认菜单和 Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 调整顶部留白 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 搜索框美化 */
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 10px;
    }

    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }

    /* 按钮美化 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar: 配置中心
# ==========================================
with st.sidebar:
    st.markdown("## Configuration")

    st.divider()

    st.markdown("#### Semantic Scholar")
    s2_api_key = st.text_input("API Key", type="password", help="Optional but recommended for speed.")

    st.divider()

    st.markdown("#### Embedding Strategy")
    embed_mode = st.radio(
        "Source",
        ["S2 Embeddings (Fast)", "Local Embeddings (Deep)"],
        label_visibility="collapsed",
        captions=[
            "Uses API vectors. Fast, but may miss papers.",
            "Computes locally. Slower, but 100% coverage."
        ]
    )
    mode_key = 's2' if "S2" in embed_mode else 'local'

    st.divider()

    st.markdown("#### LLM Intelligence")
    llm_provider = st.selectbox("Provider", ["openai", "openai-compatible", "anthropic", "gemini", "azure"])
    llm_key = st.text_input("API Key", type="password")

    llm_base_url = None
    if llm_provider == "openai-compatible":
        llm_base_url = st.text_input("Base URL", placeholder="https://api.deepseek.com")

    llm_model = st.text_input("Model Name", value="gpt-3.5-turbo")

    st.markdown("---")
    badge(type="github", name="streamlit/streamlit", url="https://github.com/streamlit/streamlit")
    st.caption("v1.0.0 | AI Innovation Contest")

# ==========================================
# Main: 主界面
# ==========================================

# Hero Section
colored_header(
    label="DeepSurvey: AI Domain Explorer",
    description="Visualize research trends, discover hidden clusters, and generate insights in seconds.",
    color_name="violet-70"
)

# 搜索与操作区
col_search, col_btn = st.columns([4, 1])
with col_search:
    query = st.text_input(
        "Search Topic",
        placeholder="e.g. 'Large Language Models' or 'Quantum Error Correction'...",
        label_visibility="collapsed"
    )

with col_btn:
    # 为了对齐输入框，加个空行
    # st.write("")
    # st.write("")
    analyze_btn = st.button("🚀 Analyze", type="primary")

# Session State 初始化
if 'data' not in st.session_state:
    st.session_state.data = None

# --- 核心逻辑执行 ---
if query and analyze_btn:
    # 使用 Status 容器替代 Spinner，看起来更高级
    with st.status(f"🕵️‍♂️ Scouting knowledge graph for: **{query}**", expanded=True) as status:

        # 1. Fetch
        st.write("📡 Connecting to Semantic Scholar Graph...")
        fetcher = PaperFetcher(api_key=s2_api_key if s2_api_key else None)
        papers = fetcher.search_papers(query, limit=100)

        if not papers:
            status.update(label="No papers found!", state="error", expanded=False)
            st.error("No papers found. Try a broader keyword.")
        else:
            st.write(f"📦 Retrieved {len(papers)} unique papers. Processing embeddings...")
            df = pd.DataFrame(papers)

            # 2. Process
            processor = DataProcessor()
            df_processed = processor.process_data(df, embedding_mode=mode_key)

            if df_processed.empty:
                status.update(label="Data Error", state="error")
                st.error("All papers dropped. Try 'Local Embeddings'.")
            else:
                st.write("🧠 Performing HDBSCAN clustering & UMAP projection...")

                # 3. LLM
                if llm_key:
                    st.write("🤖 Invoking LLM for topic summarization...")
                    summarizer = LLMSummarizer(api_key=llm_key, model_name=llm_model, base_url=llm_base_url)
                    cluster_labels = summarizer.summarize_clusters(df_processed)
                else:
                    st.warning("Skipping LLM summary (No Key). Using generic labels.")
                    cluster_labels = {i: f"Cluster {i}" for i in df_processed['cluster'].unique()}
                    if -1 in cluster_labels: cluster_labels[-1] = "Outliers"

                st.session_state.data = (df_processed, cluster_labels)
                status.update(label="Analysis Complete!", state="complete", expanded=False)

# ==========================================
# Visualization: 结果展示
# ==========================================
if st.session_state.data:
    raw_df, labels = st.session_state.data

    st.divider()

    # 1. 仪表盘统计卡片 (Metric Cards)
    m1, m2, m3, m4 = st.columns(4)
    num_clusters = len([k for k in labels.keys() if k != -1])

    m1.metric(label="Papers Analyzed", value=len(raw_df))
    m2.metric(label="Topics Identified", value=num_clusters)
    m3.metric(label="Avg. Citations", value=int(raw_df['citations'].mean()))
    m4.metric(label="Time Span", value=f"{raw_df['year'].min()} - {raw_df['year'].max()}")

    # 美化卡片样式
    style_metric_cards(border_left_color="#764ba2", box_shadow=True)

    # 2. 交互式筛选区
    st.write("")
    with st.expander("🌪️ **Filter & Explore Control Panel**", expanded=True):
        df_display = raw_df.copy()
        df_display['topic_name'] = df_display['cluster'].map(labels)

        f1, f2, f3 = st.columns(3)
        with f1:
            all_topics = sorted(list(labels.values()))
            selected_topics = st.multiselect("Select Topics", options=all_topics, default=all_topics)
        with f2:
            max_cite = int(df_display['citations'].max())
            min_cite_val = int(df_display['citations'].min())
            min_cite = st.slider("Min Citations", min_cite_val, max_cite,
                                 min_cite_val) if max_cite > min_cite_val else min_cite_val
        with f3:
            min_y, max_y = int(df_display['year'].min()), int(df_display['year'].max())
            sel_years = st.slider("Time Period", min_y, max_y, (min_y, max_y)) if max_y > min_y else (min_y, max_y)

        # 筛选逻辑
        mask = (
                (df_display['topic_name'].isin(selected_topics)) &
                (df_display['citations'] >= min_cite) &
                (df_display['year'] >= sel_years[0]) &
                (df_display['year'] <= sel_years[1])
        )
        df_filtered = df_display[mask]

    # 3. 可视化图表
    if not df_filtered.empty:
        st.subheader("🗺️ Knowledge Landscape")

        # 修复 Warning 的关键点: 使用 width="stretch" 而不是 use_container_width
        st.plotly_chart(
            plot_paper_map(df_filtered, labels),
            width="stretch"  # <--- 修复处
        )
    else:
        st.warning("No papers match your current filters.")

    # 4. 数据详情
    st.subheader("📄 Paper Details")
    st.dataframe(
        df_filtered[['title', 'topic_name', 'venue', 'year', 'citations', 'url']],
        column_config={
            "url": st.column_config.LinkColumn("Link"),
            "citations": st.column_config.ProgressColumn("Impact", format="%d", min_value=0, max_value=max_cite)
        },
        use_container_width=True  # Dataframe 这里的参数暂时还没过期，或者也可以不加
    )
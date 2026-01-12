from litellm import completion
import pandas as pd

class LLMSummarizer:
    def __init__(self, api_key, model_name="gpt-3.5-turbo", base_url=None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def summarize_clusters(self, df):
        """
        对每个聚类生成简短、独特的标签
        """
        cluster_summaries = {}
        unique_clusters = sorted(df['cluster'].unique())

        # 准备上下文：把所有聚类的代表性标题都发给 LLM，让它看到全局视角
        global_context = ""
        for c in unique_clusters:
            if c == -1: continue
            titles = df[df['cluster'] == c]['title'].head(3).tolist()
            global_context += f"Cluster {c} examples: {titles}\n"

        print(f"🤖 Summarizing {len(unique_clusters)} topics with LLM: {self.model_name}...")

        # -----------------------------------------------------
        # 关键修复：处理 OpenAI Compatible 的模型名称
        # 如果提供了 base_url，且模型名不包含 provider 前缀，强制加上 'openai/'
        # 这告诉 litellm 使用 OpenAI 的协议格式发送请求
        # -----------------------------------------------------
        target_model = self.model_name
        if self.base_url and "/" not in target_model:
            target_model = f"openai/{target_model}"

        for c in unique_clusters:
            if c == -1:
                cluster_summaries[c] = "Outliers / Noise"
                continue
            # 取出该类中引用最高的 5 篇作为代表
            cluster_papers = df[df['cluster'] == c].sort_values(by='citations', ascending=False).head(5)
            paper_titles = cluster_papers['title'].tolist()

            prompt = f"""
            You are a senior researcher analyzing a map of academic papers.
            Here is the global context of all clusters found:
            {global_context}

            Now, generate a specific, technical label for Cluster {c}.
            The papers in Cluster {c} are:
            {paper_titles}

            Constraint:
            1. Label must be DISTINCT from other clusters.
            2. Max 5 words.
            3. Do not use generic words like "Research" or "Analysis".
            4. Output ONLY the label string.
            """

            try:
                response = completion(
                    model=target_model,  # 使用带前缀的模型名
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                label = response.choices[0].message.content.strip().replace('"', '')
                cluster_summaries[c] = label
                print(f"  - Cluster {c}: {label}")  # 打印进度，让你看到它在工作
            except Exception as e:
                cluster_summaries[c] = f"Topic {c}"
                print(f"❌ LLM Error for Cluster {c}: {e}")

        return cluster_summaries
import semanticscholar
from semanticscholar import SemanticScholar
import re

class PaperFetcher:
    def __init__(self, api_key=None):
        self.sch = SemanticScholar(api_key=api_key)

    def search_papers(self, query, limit=100):
        """
        Keyword 搜索 + 增强的数据清洗
        """
        # 请求更多字段以修复 Venue 问题
        fields = [
            'title', 'abstract', 'year', 'citationCount',
            'venue', 'publicationVenue', 'externalIds',
            'url', 'publicationTypes', 'embedding'
        ]

        try:
            print(f"🔍 Searching S2 for: {query}...")
            results = self.sch.search_paper(query, limit=limit, fields=fields)

            papers_data = []
            for item in results:
                if not item.title or not item.year:
                    continue

                # --- 修复 Venue: 优先取 publicationVenue.name ---
                venue_name = "Unknown"
                if item.publicationVenue and isinstance(item.publicationVenue, dict):
                    venue_name = item.publicationVenue.get('name', "Unknown")
                elif item.venue:
                    venue_name = item.venue

                # 构建链接
                link = item.url
                if item.externalIds:
                    if 'DOI' in item.externalIds:
                        link = f"https://doi.org/{item.externalIds['DOI']}"
                    elif 'ArXiv' in item.externalIds:
                        link = f"https://arxiv.org/abs/{item.externalIds['ArXiv']}"

                # 来源评分
                source_score = 1
                if item.publicationTypes and 'Journal' in item.publicationTypes:
                    source_score = 2
                elif venue_name and 'arxiv' not in venue_name.lower() and venue_name != "Unknown":
                    source_score = 2

                papers_data.append({
                    'paper_id': item.paperId,
                    'title': item.title,
                    'abstract': item.abstract,
                    'year': int(item.year),
                    'citations': item.citationCount or 0,
                    'venue': venue_name,
                    'url': link,
                    'embedding': item.embedding['vector'] if item.embedding else None,
                    'source_score': source_score
                })

            return self._deduplicate(papers_data)

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return []

    def _deduplicate(self, papers):
        unique_map = {}
        for p in papers:
            # 极简标题去重 (去标点、空格、转小写)
            norm_title = re.sub(r'[^a-z0-9]', '', p['title'].lower())

            if norm_title not in unique_map:
                unique_map[norm_title] = p
            else:
                existing = unique_map[norm_title]
                if p['source_score'] > existing['source_score']:
                    unique_map[norm_title] = p
                elif p['source_score'] == existing['source_score']:
                    if p['citations'] > existing['citations']:
                        unique_map[norm_title] = p

        print(f"🧹 Deduplication: {len(papers)} -> {len(unique_map)}")
        return list(unique_map.values())
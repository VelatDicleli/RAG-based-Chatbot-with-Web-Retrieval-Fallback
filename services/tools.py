import json
from langchain_community.tools import WikipediaQueryRun, BraveSearch
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from config.settings import BRAVE_API_KEY
from core.logger import setup_logger

logger = setup_logger(__name__)

class SearchTools:
    """Class for handling external search tools."""
    
    def __init__(self):
        """Initialize search tools."""
        try:
            logger.info("Initializing search tools")
            self.tavily = TavilySearchResults(
                max_results=5,
                include_answer=True,
                include_raw_content=True,
            )
            
            self.wikipedia = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="tr")
            )
            
            self.brave = BraveSearch.from_api_key(
                api_key=BRAVE_API_KEY,
                search_kwargs={"count": 3}
            )
        except Exception as e:
            logger.error(f"Error initializing search tools: {str(e)}")
            raise
    
    def search_with_brave(self, query):
        """Perform search using Brave."""
        try:
            logger.info(f"Performing Brave search for: {query}")
            result = self.brave.run(query)
            if result:
                br_parsed = json.loads(result)
                snippets = []
                for item in br_parsed:
                    if "snippet" in item and item["snippet"]:
                        snippets.append(f'Web Araması Sonucu:\n{item["snippet"]}')
                return snippets
            return []
        except Exception as e:
            logger.error(f"Brave search error: {str(e)}")
            return []
    
    def search_with_tavily(self, query):
        """Perform search using Tavily."""
        try:
            logger.info(f"Performing Tavily search for: {query}")
            result = self.tavily.invoke(query, k=4)
            if result:
                snippets = []
                for item in result:
                    if "content" in item and item["content"]:
                        snippets.append(f'Web Araması Sonucu:\n{item["content"]}')
                return snippets
            return []
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return []
    
    def search_with_wikipedia(self, query):
        """Perform search using Wikipedia."""
        try:
            logger.info(f"Performing Wikipedia search for: {query}")
            result = self.wikipedia.run(query)
            if result:
                return [f"Wikipedia Sonucu:\n{result}"]
            return []
        except Exception as e:
            logger.error(f"Wikipedia search error: {str(e)}")
            return []
    
    def search_all(self, query):
        """Search using all available tools and return combined results."""
        all_results = []
        
        # Try Brave search
        brave_results = self.search_with_brave(query)
        if brave_results:
            all_results.extend(brave_results)
        
    
        if not all_results:
            tavily_results = self.search_with_tavily(query)
            if tavily_results:
                all_results.extend(tavily_results)
        
       
        if not all_results:
            wiki_results = self.search_with_wikipedia(query)
            if wiki_results:
                all_results.extend(wiki_results)
        
        if all_results:
            return "\n\n".join(all_results)
        return "Dış kaynaklarda ilgili bilgi bulunamadı."
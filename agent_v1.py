import os
import re
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.output_parsers.retry import RetryOutputParser
from langchain_openai import ChatOpenAI

from tools.search_engines import (FileProcessor, RobustTextExtractor,
                                  TabularDataAnalyzer, WebSearchTool,
                                  WikipediaSearchTool)

# Load environment variables (expects OPENROUTER_API_KEY to be set)
load_dotenv()


class GaiaAgent:
    """Enhanced GAIA agent with interactive mode and improved tool integration"""

    def __init__(
        self,
        openrouter_key: str,
        model_name: str = "tngtech/deepseek-r1t-chimera:free",
        system_prompt: str = None,
    ):
        self._configure_warnings()
        self.openrouter_key = openrouter_key
        self.model_name = model_name

        # Initialize components
        self.tools = self._initialize_default_tools()
        self.memory = self._configure_memory()
        self.llm = self._initialize_llm()
        self.agent_executor = self._setup_agent()

    def _configure_warnings(self):
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def _initialize_default_tools(self) -> List[Tool]:
        return [
            self._create_wikipedia_tool(),
            self._create_web_search_tool(),
            self._create_file_tool(),
            self._create_image_tool(),
            self._create_table_tool(),
        ]

    def _create_wikipedia_tool(self) -> Tool:
        return Tool(
            name="wikipedia_search",
            func=self._search_wikipedia,
            description=(
                "Search Wikipedia for factual information. "
                "Input should be a clear search query."
            ),
        )

    def _create_web_search_tool(self) -> Tool:
        return Tool(
            name="web_search",
            func=self._web_search,
            description=(
                "Search the web for current information. "
                "Use 'site:domain.com' to filter by domain. "
                "Always prefer this for recent/controversial topics."
            ),
        )

    def _create_file_tool(self) -> Tool:
        return Tool(
            name="file_processor",
            func=self._process_file,
            description="Handle file operations. Use 'content||filename' to save or filename to read.",
        )

    def _create_image_tool(self) -> Tool:
        return Tool(
            name="image_analysis",
            func=self._analyze_image,
            description="Analyze images using OCR and AI. Input is image path.",
        )

    def _create_table_tool(self) -> Tool:
        return Tool(
            name="tabular_analysis",
            func=self._analyze_table,
            description="Analyze CSV/Excel files. Use 'file_path||query' format.",
        )

    def _search_wikipedia(self, query: str) -> str:
        try:
            return WikipediaSearchTool().search(query)
        except Exception as e:
            return f"Wikipedia search failed: {str(e)}"

    def _web_search(self, query: str) -> str:
        try:
            domain = None
            web_search = WebSearchTool()

            if "site:" in query:
                query, domain = query.split("site:", 1)

            search_result = web_search.search(query.strip())
            if domain:
                search_result += f"\nDomain filter: {domain.strip()}"
            return search_result
        except Exception as e:
            return f"Web search failed: {str(e)}"

    def _process_file(self, input_str: str) -> str:
        """Internal file processor with input parsing"""
        try:
            if "||" in input_str:
                content, filename = input_str.split("||", 1)
                file_processor = FileProcessor()
                result = file_processor.save_content(content.strip(), filename.strip())
                return result
            else:
                return FileProcessor().read_file(input_str.strip())
        except Exception as e:
            return f"File processing error: {str(e)}"

    def _analyze_image(self, image_path: str) -> str:
        try:
            extractor = RobustTextExtractor(self.openrouter_key)
            return extractor.get_text_from_image(image_path)
        except Exception as e:
            return f"Image analysis failed: {str(e)}"

    def _analyze_table(self, input_str: str) -> str:
        """Internal tabular data analyzer with query support"""
        try:
            if "||" in input_str:
                file_path, query = input_str.split("||", 1)
            else:
                file_path, query = input_str, None
            return TabularDataAnalyzer().analyze(file_path.strip(), query)
        except Exception as e:
            return f"Tabular analysis failed: {str(e)}"

    def _configure_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )

    def _initialize_llm(self):
        """Initialize the LLM with proper OpenRouter configuration"""
        return ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_key,
            model_name=self.model_name,
            temperature=0.2,
            max_tokens=2000,
        )

    def _setup_agent(self) -> AgentExecutor:
        """Configure agent with robust output parsing via RetryOutputParser."""
        from langchain.agents import initialize_agent

        PREFIX = """You are GAIA - a sophisticated AI assistant with these capabilities:
1. Web search (general and domain-specific)
2. Image text extraction and analysis
3. Tabular data processing

TOOLS:
------"""

        FORMAT_INSTRUCTIONS = """Always respond using this format:
Thought: Consider the question and available tools
Action: Select a tool from [{tool_names}]
Action Input: Prepare the tool's input
Observation: Analyze the tool's output
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I know the final answer now
Final Answer: Present the verified information"""

        SUFFIX = """Conversation History:
{chat_history}
New Input: {input}
{agent_scratchpad}"""

        # Base ReAct parser
        base_parser = ReActSingleInputOutputParser()

        # Wrap it in a retrying parser to handle minor format deviations
        retry_parser = RetryOutputParser.from_llm(
            parser=base_parser,
            llm=self.llm,
            max_retries=3,
        )

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": PREFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
                "suffix": SUFFIX,
                "input_variables": ["input", "chat_history", "agent_scratchpad"],
            },
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            output_parser=retry_parser,
        )

    def _get_recovery_suggestion(self, error: str) -> str:
        """Generate context-aware recovery suggestions"""
        if "404" in error:
            return "Verify the requested resource exists and is accessible."
        if "timeout" in error.lower():
            return "Try simplifying the query or breaking it into smaller parts."
        return "Consider rephrasing the query or checking input formats."

    def query(self, input_text: str) -> Dict[str, Any]:
        """Execute query with enhanced error handling and retries"""
        try:
            response = self.agent_executor({"input": input_text})
            return {
                "success": True,
                "output": self._clean_response(response["output"]),
                "metadata": {
                    "steps": len(response.get("intermediate_steps", [])),
                    "model": self.model_name,
                    "timestamp": time.time(),
                    "tools_used": [
                        action.tool
                        for action, _ in response.get("intermediate_steps", [])
                    ],
                },
            }
        except Exception as e:
            print(e)
            return {
                "success": False,
                "error": str(e),
                "recovery_suggestion": self._get_recovery_suggestion(str(e)),
            }

    def _clean_response(self, raw_response: str) -> str:
        """Clean agent output with regex patterns"""
        patterns = [
            r"(> Entering new AgentExecutor chain...|> Finished chain\.)",
            r"Thought:.*?Action:.*?Action Input:.*?Observation:.*?\n",
            r"\x1b\[.*?m",  # ANSI escape codes
        ]
        for pattern in patterns:
            raw_response = re.sub(pattern, "", raw_response, flags=re.DOTALL)
        return raw_response.strip()

    def run_interactive(self):
        print("GAIA Assistant Ready (Type 'exit' to quit)\n")
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                response = self.query(query)
                print(f"\nAssistant: {response['output']}\n")

            except KeyboardInterrupt:
                print("\nSession ended.")
                break

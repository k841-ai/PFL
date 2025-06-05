from typing import Dict, List, Any
import logging
from datetime import datetime
from graph_utils import GraphGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class QueryAgent(BaseAgent):
    def __init__(self, rag_engine):
        super().__init__("query")
        self.rag_engine = rag_engine
        self.agent_type = "text"
        self.supported_queries = [
            "what", "how", "when", "where", "why", "who",
            "show", "tell", "explain", "describe", "list"
        ]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "").lower()
        
        # Check if query is text-based
        if any(term in query for term in self.supported_queries):
            self.logger.info(f"Query agent processing: {query}")
            
            # Get context and generate response using RAG engine
            context = self.rag_engine._get_relevant_context(query)
            response = self.rag_engine._generate_response(query, context)
            
            return {
                "agent": self.name,
                "agent_type": self.agent_type,
                "confidence": 0.9,
                "response": response,
                "context_used": context
            }
        return None

class GraphAgent(BaseAgent):
    def __init__(self, rag_engine):
        super().__init__("graph")
        self.rag_engine = rag_engine
        self.agent_type = "visualization"
        self.graph_generator = GraphGenerator()
        self.supported_queries = [
            "graph", "chart", "plot", "visualize", "trend",
            "compare", "relationship", "correlation", "analysis",
            "show me", "display", "illustrate"
        ]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "").lower()
        
        # Check if query is graph-related
        if any(term in query for term in self.supported_queries):
            self.logger.info(f"Graph agent processing: {query}")
            
            try:
                # Get context from RAG engine
                context_docs = self.rag_engine._get_relevant_context(query)
                
                # Convert Document objects to strings and log them
                context = []
                for doc in context_docs:
                    content = doc.page_content
                    self.logger.info(f"Context document: {content[:200]}...")  # Log first 200 chars
                    context.append(content)
                
                if not context:
                    return {
                        "agent": self.name,
                        "agent_type": self.agent_type,
                        "confidence": 0.0,
                        "response": "No relevant context found for graph generation.",
                        "graph_data": None,
                        "embed_code": None
                    }
                
                # Generate graph
                graph_result = self.graph_generator.generate_graph(query, context)
                
                if "error" in graph_result:
                    self.logger.error(f"Graph generation error: {graph_result['error']}")
                    return {
                        "agent": self.name,
                        "agent_type": self.agent_type,
                        "confidence": 0.0,
                        "response": f"Error generating graph: {graph_result['error']}",
                        "graph_data": None,
                        "embed_code": None
                    }
                
                self.logger.info(f"Graph generated successfully: {graph_result['graph_type']}")
                
                return {
                    "agent": self.name,
                    "agent_type": self.agent_type,
                    "confidence": 0.9,
                    "response": f"Generated {graph_result['graph_type']} graph for your query",
                    "graph_data": graph_result["graph_data"],
                    "embed_code": graph_result["embed_code"],
                    "filepath": graph_result["filepath"],
                    "context_used": context  # Add context to response
                }
            except Exception as e:
                self.logger.error(f"Error in graph generation: {str(e)}")
                return {
                    "agent": self.name,
                    "agent_type": self.agent_type,
                    "confidence": 0.0,
                    "response": f"Error processing graph request: {str(e)}",
                    "graph_data": None,
                    "embed_code": None
                }
        return None

class MasterAgent:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.agents = [
            QueryAgent(rag_engine),
            GraphAgent(rag_engine)
        ]
        self.logger = logging.getLogger("agent.master")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the master agent and specialized agents.
        Returns a response with agent attribution and confidence scores.
        """
        start_time = datetime.now()
        
        # Prepare input for agents
        input_data = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }

        # Get responses from all agents
        agent_responses = []
        for agent in self.agents:
            try:
                response = agent.process(input_data)
                if response:
                    agent_responses.append(response)
            except Exception as e:
                self.logger.error(f"Error in agent {agent.name}: {str(e)}")

        # Select best response or combine responses
        final_response = self._select_best_response(agent_responses)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": final_response["response"],
            "agent_used": final_response["agent"],
            "agent_type": final_response["agent_type"],
            "confidence": final_response["confidence"],
            "context_used": final_response.get("context_used", []),
            "graph_data": final_response.get("graph_data", {}),
            "embed_code": final_response.get("embed_code"),
            "filepath": final_response.get("filepath"),
            "processing_time": processing_time,
            "status": "success"
        }

    def _select_best_response(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best response based on confidence scores.
        In the future, this could be more sophisticated with ensemble methods.
        """
        if not responses:
            return {
                "agent": "master",
                "agent_type": "fallback",
                "confidence": 0.0,
                "response": "I couldn't find a suitable response for your query."
            }
        
        # Sort by confidence and return the highest
        return sorted(responses, key=lambda x: x["confidence"], reverse=True)[0] 
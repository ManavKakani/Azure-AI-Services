import os
from opentelemetry import trace
import time
import random
import dotenv
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import json
from opentelemetry.metrics import get_meter
from opentelemetry.trace.status import Status, StatusCode

dotenv.load_dotenv()

# Install opentelemetry with command "pip install opentelemetry-sdk".
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, 
    UserMessage, 
    CompletionsFinishReason,
    ToolMessage,
    AssistantMessage,
    ChatCompletionsToolCall,
    ChatCompletionsToolDefinition,
    FunctionDefinition,
)
from azure.core.credentials import AzureKeyCredential

# Configure Azure Monitor OpenTelemetry
from azure.monitor.opentelemetry import configure_azure_monitor

# [START trace_setting]
from azure.core.settings import settings
settings.tracing_implementation = "opentelemetry"
# [END trace_setting]

# Enable content recording for generative AI spans
os.environ['AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED'] = 'true'

# Setup Azure Monitor OpenTelemetry exporter
connection_string = os.getenv("AZURE_MONITOR_CONNECTION_STRING")
configure_azure_monitor(
    connection_string=connection_string,
    enable_traces=True,
    enable_metrics=True
)

# Create tracer and meter
tracer = trace.get_tracer(__name__)
meter = get_meter(__name__)

# Create metric instruments
response_time_histogram = meter.create_histogram(
    name="chat_response_time",
    description="Time taken for chat responses",
    unit="ms"
)

evaluation_gauge = meter.create_gauge(
    name="chat_evaluation_metrics",
    description="Evaluation metrics for chat responses",
    unit="score"
)

@dataclass
class Document:
    content: str
    metadata: Dict
    embedding: np.ndarray = None

class DocumentProcessor:
    def __init__(self):
        self.documents = [
            Document(
                content="""Seattle Weather Patterns:
                Seattle has a temperate climate with mild winters and cool summers.
                Average summer temperatures range from 70-75°F (21-24°C).
                The city experiences frequent cloud cover and light rain.""",
                metadata={"source": "weather_database", "city": "Seattle", "doc_type": "climate_info", "last_updated": "2024"}
            ),
            Document(
                content="""New York City Climate:
                NYC has a humid subtropical climate with hot summers and cold winters.
                Summer temperatures typically range from 78-85°F (26-29°C).
                The city experiences all four seasons distinctly.""",
                metadata={"source": "weather_database", "city": "New York City", "doc_type": "climate_info", "last_updated": "2024"}
            ),
            Document(
                content="""Seattle Historical Data 2023:
                Average temperature: 75°F
                Rainfall: Light to moderate
                Air quality index: Good (AQI 45)""",
                metadata={"source": "historical_data", "city": "Seattle", "doc_type": "historical_data", "last_updated": "2023"}
            ),
            Document(
                content="""New York City Historical Data 2023:
                Average temperature: 80°F
                Rainfall: Moderate
                Air quality index: Moderate (AQI 65)""",
                metadata={"source": "historical_data", "city": "New York City", "doc_type": "historical_data", "last_updated": "2023"}
            )
        ]

    def search_documents(self, query: str, n_results: int = 2) -> List[Document]:
        matching_docs = []
        query_terms = query.lower().split()
        
        for doc in self.documents:
            score = sum(1 for term in query_terms if term in doc.content.lower())
            if score > 0:
                matching_docs.append((score, doc))
        
        matching_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in matching_docs[:n_results]]

@tracer.start_as_current_span("get_temperature")
def get_temperature(city: str) -> str:
    span = trace.get_current_span()
    span.set_attribute("requested_city", city)

    if city == "Seattle":
        return "75"
    elif city == "New York City":
        return "80"
    else:
        return "Unavailable"

def get_weather(city: str) -> str:
    if city == "Seattle":
        return "Nice weather"
    elif city == "New York City":
        return "Good weather"
    else:
        return "Unavailable"

def chat_completion_with_function_call(key, endpoint):
    session_id = f"session_{random.randint(1000, 9999)}"
    doc_processor = DocumentProcessor()
    
    with tracer.start_as_current_span("chat_session") as session_span:
        try:
            session_span.set_attribute("session_id", session_id)
            session_start_time = time.time()
            
            # Change add to set for gauge metrics
            evaluation_gauge.set(1, {"metric_name": "session_start", "session_id": session_id})
            
            conversations = [
                "What is the weather and temperature in Seattle?",
                "How about New York City?",
                "Compare the temperatures of both cities."
            ]

            # Initialize client and tool definitions
            client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
            
            weather_description = ChatCompletionsToolDefinition(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Returns description of the weather in the specified city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city for which weather info is requested",
                            },
                        },
                        "required": ["city"],
                    }
                )
            )

            temperature_in_city = ChatCompletionsToolDefinition(
                function=FunctionDefinition(
                    name="get_temperature",
                    description="Returns the current temperature for the specified city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city for which temperature info is requested",
                            },
                        },
                        "required": ["city"],
                    }
                )
            )
            
            for turn_idx, user_query in enumerate(conversations):
                with tracer.start_as_current_span("conversation_turn") as turn_span:
                    turn_start = time.time()
                    
                    turn_span.set_attribute("turn_number", turn_idx + 1)
                    turn_span.set_attribute("user_query", user_query)
                    turn_span.set_attribute("session_id", session_id)
                    
                    # Document processing with metrics
                    with tracer.start_as_current_span("document_processing") as doc_span:
                        doc_start = time.time()
                        relevant_docs = doc_processor.search_documents(user_query)
                        doc_processing_time = (time.time() - doc_start) * 1000
                        
                        doc_span.set_attribute("processing_time_ms", doc_processing_time)
                        doc_span.set_attribute("num_docs_processed", len(relevant_docs))
                        response_time_histogram.record(
                            doc_processing_time,
                            {"operation": "document_processing", "session_id": session_id}
                        )
                        
                        for idx, doc in enumerate(relevant_docs):
                            doc_span.set_attribute(f"doc_{idx}_source", doc.metadata.get("source", "unknown"))
                            doc_span.set_attribute(f"doc_{idx}_doc_type", doc.metadata.get("doc_type", "unknown"))
                        
                        context = "\n".join([doc.content for doc in relevant_docs])

                    # Update messages with context
                    messages = [
                        SystemMessage(f"""You are a helpful assistant. Use the following context to answer questions:
                        
                        Context:
                        {context}
                        
                        Answer based on this context when possible."""),
                        UserMessage(user_query),
                    ]

                    # API call with metrics
                    with tracer.start_as_current_span("api_call") as api_span:
                        api_start = time.time()
                        response = client.complete(messages=messages, tools=[weather_description, temperature_in_city])
                        api_latency = (time.time() - api_start) * 1000
                        
                        api_span.set_attribute("api_latency_ms", api_latency)
                        response_time_histogram.record(
                            api_latency,
                            {"operation": "api_call", "session_id": session_id}
                        )

                    # Process function calls if any
                    if response.choices[0].finish_reason == CompletionsFinishReason.TOOL_CALLS:
                        with tracer.start_as_current_span("tool_calls_processing") as tool_span:
                            tool_start = time.time()
                            
                            if response.choices[0].message.tool_calls is not None:
                                messages.append(AssistantMessage(tool_calls=response.choices[0].message.tool_calls))
                                
                                for tool_call in response.choices[0].message.tool_calls:
                                    if isinstance(tool_call, ChatCompletionsToolCall):
                                        function_args = json.loads(tool_call.function.arguments.replace("'", '"'))
                                        tool_span.set_attribute("function_name", tool_call.function.name)
                                        tool_span.set_attribute("function_args", str(function_args))
                                        
                                        callable_func = globals()[tool_call.function.name]
                                        function_response = callable_func(**function_args)
                                        messages.append(ToolMessage(function_response, tool_call_id=tool_call.id))
                                
                                response = client.complete(messages=messages, tools=[weather_description, temperature_in_city])
                            
                            tool_processing_time = (time.time() - tool_start) * 1000
                            tool_span.set_attribute("processing_time_ms", tool_processing_time)
                            response_time_histogram.record(
                                tool_processing_time,
                                {"operation": "tool_processing", "session_id": session_id}
                            )

                    # Update the metrics section in the conversation loop
                    metrics = {
                        "coherence": random.randint(3, 5),
                        "groundedness": random.randint(3, 5),
                        "relevance": random.randint(3, 5),
                        "user_feedback": random.choice([-1, 0, 1])
                    }
                    
                    for metric_name, metric_value in metrics.items():
                        turn_span.set_attribute(metric_name, metric_value)
                        evaluation_gauge.set(
                            metric_value,
                            {
                                "metric_name": metric_name,
                                "turn_number": turn_idx + 1,
                                "session_id": session_id
                            }
                        )
                    
                    # Turn duration metrics
                    turn_duration = (time.time() - turn_start) * 1000
                    turn_span.set_attribute("turn_duration_ms", turn_duration)
                    response_time_histogram.record(
                        turn_duration,
                        {"operation": "turn_complete", "session_id": session_id}
                    )
                    
                    print(f"Turn {turn_idx + 1} Response: {response.choices[0].message.content}")
                    turn_span.set_status(Status(StatusCode.OK))
            
            # Session completion metrics
            session_duration = (time.time() - session_start_time) * 1000
            session_span.set_attribute("total_turns", len(conversations))
            session_span.set_attribute("session_duration_ms", session_duration)
            response_time_histogram.record(
                session_duration,
                {"operation": "session_complete", "session_id": session_id}
            )
            
            session_span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            session_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

def main():
    try:
        endpoint = os.getenv("AZURE_AI_CHAT_ENDPOINT")
        key = os.getenv("AZURE_AI_CHAT_KEY")
        if not endpoint or not key:
            raise ValueError("AZURE_AI_CHAT_ENDPOINT and AZURE_AI_CHAT_KEY environment variables are required")
        
        # Instrument AI Inference
        from azure.ai.inference.tracing import AIInferenceInstrumentor
        AIInferenceInstrumentor().instrument()
        
        chat_completion_with_function_call(key, endpoint)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        AIInferenceInstrumentor().uninstrument()

if __name__ == "__main__":
    main()

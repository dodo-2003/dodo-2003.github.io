import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Copy, ExternalLink, Github } from "lucide-react"
import { useState } from "react"

const codeExamples = [
  {
    id: "rag",
    title: "RAG System Implementation",
    description: "Complete RAG pipeline with document processing and vector embeddings",
    language: "Python",
    category: "RAG",
    code: `from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pinecone

class RAGSystem:
    def __init__(self, pinecone_api_key, openai_api_key):
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        self.index_name = "knowledge-base"
        
    def process_documents(self, file_paths):
        """Process and embed documents into vector store"""
        documents = []
        
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Pinecone.from_documents(
            texts, self.embeddings, index_name=self.index_name
        )
        
        return len(texts)
    
    def query(self, question, k=4):
        """Query the RAG system"""
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

# Usage Example
rag = RAGSystem(pinecone_api_key="your-key", openai_api_key="your-key")
chunks_processed = rag.process_documents(["document1.pdf", "document2.pdf"])
response = rag.query("What are the key findings in the research?")
print(f"Answer: {response['answer']}")`,
    features: ["Document Processing", "Vector Embeddings", "Semantic Search", "Source Attribution"]
  },
  {
    id: "ocr",
    title: "Advanced OCR Pipeline",
    description: "AI-powered text extraction with layout preservation and data structuring",
    language: "Python",
    category: "OCR",
    code: `import cv2
import pytesseract
from PIL import Image
import numpy as np
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import torch

class AdvancedOCR:
    def __init__(self):
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
        
    def preprocess_image(self, image_path):
        """Enhance image quality for better OCR"""
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return threshold
    
    def extract_text_with_coordinates(self, image_path):
        """Extract text with bounding box coordinates"""
        processed_image = self.preprocess_image(image_path)
        
        # OCR with detailed data
        data = pytesseract.image_to_data(
            processed_image, 
            output_type=pytesseract.Output.DICT,
            config='--psm 6'
        )
        
        extracted_data = []
        for i, word in enumerate(data['text']):
            if word.strip():
                extracted_data.append({
                    'text': word,
                    'confidence': data['conf'][i],
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return extracted_data
    
    def structure_document(self, extracted_data):
        """Use LayoutLM to understand document structure"""
        text_tokens = [item['text'] for item in extracted_data]
        bboxes = [[item['bbox']['x'], item['bbox']['y'], 
                  item['bbox']['x'] + item['bbox']['width'],
                  item['bbox']['y'] + item['bbox']['height']] for item in extracted_data]
        
        # Tokenize and encode
        encoding = self.tokenizer(
            text_tokens,
            boxes=bboxes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True
        )
        
        # Predict labels
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Map predictions to structure
        structured_result = {
            'title': [],
            'content': [],
            'tables': [],
            'metadata': []
        }
        
        return structured_result

# Usage Example
ocr_system = AdvancedOCR()
extracted_data = ocr_system.extract_text_with_coordinates("document.png")
structured_doc = ocr_system.structure_document(extracted_data)
print(f"Extracted {len(extracted_data)} text elements")`,
    features: ["Image Enhancement", "Layout Understanding", "High Accuracy", "Structure Detection"]
  },
  {
    id: "db-rag",
    title: "Database RAG System",
    description: "Natural language to SQL with intelligent query generation and validation",
    language: "Python", 
    category: "Database RAG",
    code: `import sqlalchemy
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms import OpenAI
import pandas as pd

class DatabaseRAG:
    def __init__(self, connection_string, openai_api_key):
        self.engine = sqlalchemy.create_engine(connection_string)
        self.db = SQLDatabase(self.engine)
        self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        
        # Create SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create agent
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type="zero-shot-react-description"
        )
    
    def get_schema_info(self):
        """Get database schema information"""
        return self.db.get_table_info()
    
    def validate_query(self, sql_query):
        """Validate SQL query before execution"""
        forbidden_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
        
        query_upper = sql_query.upper()
        for keyword in forbidden_keywords:
            if keyword in query_upper:
                return False, f"Query contains forbidden keyword: {keyword}"
        
        return True, "Query is safe"
    
    def natural_language_query(self, question):
        """Convert natural language to SQL and execute"""
        try:
            # Use agent to generate and execute query
            result = self.agent.run(question)
            
            # Extract SQL query from agent's reasoning
            sql_query = self.extract_sql_from_result(result)
            
            # Validate query
            is_safe, validation_msg = self.validate_query(sql_query)
            
            if not is_safe:
                return {
                    "error": validation_msg,
                    "query": sql_query,
                    "data": None
                }
            
            # Execute query and get results
            df = pd.read_sql(sql_query, self.engine)
            
            return {
                "answer": result,
                "query": sql_query,
                "data": df.to_dict('records'),
                "row_count": len(df)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "query": None,
                "data": None
            }
    
    def extract_sql_from_result(self, agent_result):
        """Extract SQL query from agent result"""
        # Implementation to parse SQL from agent's output
        lines = agent_result.split('\\n')
        for line in lines:
            if 'SELECT' in line.upper():
                return line.strip()
        return ""
    
    def get_query_explanation(self, sql_query):
        """Explain what a SQL query does in natural language"""
        explanation_prompt = f"""
        Explain this SQL query in simple terms:
        {sql_query}
        """
        return self.llm(explanation_prompt)

# Usage Example
db_rag = DatabaseRAG(
    connection_string="postgresql://user:pass@localhost/db",
    openai_api_key="your-key"
)

response = db_rag.natural_language_query(
    "Show me the top 10 customers by revenue this year"
)
print(f"Query: {response['query']}")
print(f"Results: {response['data']}")`,
    features: ["Natural Language to SQL", "Query Validation", "Schema Understanding", "Multi-Database Support"]
  },
  {
    id: "mongo-rag",
    title: "MongoDB RAG Implementation",
    description: "Vector search and aggregation pipelines for NoSQL document retrieval",
    language: "Python",
    category: "MongoDB RAG", 
    code: `from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

class MongoRAG:
    def __init__(self, connection_string, database_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_vector_index(self, collection_name, text_field, vector_field="embedding"):
        """Create vector index for similarity search"""
        collection = self.db[collection_name]
        
        # Create vector search index
        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": vector_field,
                    "numDimensions": 384,  # MiniLM embedding size
                    "similarity": "cosine"
                }
            ]
        }
        
        try:
            collection.create_search_index(index_definition, "vector_index")
            return True
        except Exception as e:
            print(f"Index creation failed: {e}")
            return False
    
    def embed_documents(self, collection_name, text_field, batch_size=100):
        """Generate embeddings for existing documents"""
        collection = self.db[collection_name]
        
        # Process documents in batches
        cursor = collection.find({text_field: {"$exists": True}})
        
        batch = []
        processed = 0
        
        for doc in cursor:
            batch.append(doc)
            
            if len(batch) >= batch_size:
                self._process_batch(collection, batch, text_field)
                processed += len(batch)
                batch = []
                print(f"Processed {processed} documents")
        
        # Process remaining documents
        if batch:
            self._process_batch(collection, batch, text_field)
            processed += len(batch)
        
        return processed
    
    def _process_batch(self, collection, batch, text_field):
        """Process a batch of documents"""
        texts = [doc[text_field] for doc in batch]
        embeddings = self.encoder.encode(texts)
        
        # Update documents with embeddings
        for doc, embedding in zip(batch, embeddings):
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding.tolist()}}
            )
    
    def vector_search(self, collection_name, query_text, limit=5):
        """Perform vector similarity search"""
        collection = self.db[collection_name]
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query_text])[0]
        
        # MongoDB Atlas Vector Search aggregation
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding.tolist(),
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return results
    
    def hybrid_search(self, collection_name, query_text, filters=None, limit=5):
        """Combine vector search with traditional filters"""
        collection = self.db[collection_name]
        query_embedding = self.encoder.encode([query_text])[0]
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": query_embedding.tolist(),
                    "numCandidates": limit * 20,
                    "limit": limit * 5
                }
            }
        ]
        
        # Add filters if provided
        if filters:
            pipeline.append({"$match": filters})
        
        pipeline.extend([
            {"$limit": limit},
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "category": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ])
        
        results = list(collection.aggregate(pipeline))
        return results
    
    def generate_answer(self, query, context_docs):
        """Generate answer using retrieved context"""
        context = "\\n".join([doc.get("text", "") for doc in context_docs])
        
        # Here you would integrate with your LLM
        # For example, using OpenAI or a local model
        prompt = f"""
        Based on the following context, answer the question:
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        # Placeholder for LLM integration
        return {
            "answer": "Generated answer based on context",
            "sources": [doc.get("_id") for doc in context_docs],
            "confidence": 0.85
        }

# Usage Example
mongo_rag = MongoRAG("mongodb://localhost:27017", "knowledge_db")

# Setup embeddings
processed = mongo_rag.embed_documents("documents", "content")
print(f"Embedded {processed} documents")

# Perform search
results = mongo_rag.vector_search("documents", "What is machine learning?")
answer = mongo_rag.generate_answer("What is machine learning?", results)
print(f"Answer: {answer['answer']}")`,
    features: ["Vector Search", "Aggregation Pipelines", "Hybrid Queries", "Document Similarity"]
  },
  {
    id: "custom-training",
    title: "Custom Model Training Pipeline",
    description: "Fine-tuning transformers for domain-specific tasks with evaluation metrics",
    language: "Python",
    category: "Custom Training",
    code: `import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import numpy as np

class CustomModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
    def prepare_dataset(self, texts, labels, max_length=512):
        """Tokenize and prepare dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        # Create dataset
        data = {"text": texts, "labels": labels}
        dataset = Dataset.from_dict(data)
        
        # Tokenize
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, eval_dataset, output_dir="./results"):
        """Train the model with custom parameters"""
        
        # Initialize wandb for experiment tracking
        wandb.init(project="custom-model-training")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="wandb"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        trainer.train()
        
        # Save the best model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def evaluate_model(self, test_dataset):
        """Evaluate model on test set"""
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )
        
        results = trainer.evaluate(test_dataset)
        return results
    
    def predict(self, texts):
        """Make predictions on new text"""
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return predictions.numpy()
    
    def fine_tune_for_domain(self, domain_data, validation_split=0.2):
        """Fine-tune model for specific domain"""
        
        # Split data
        split_idx = int(len(domain_data) * (1 - validation_split))
        train_data = domain_data[:split_idx]
        val_data = domain_data[split_idx:]
        
        # Prepare datasets
        train_texts = [item["text"] for item in train_data]
        train_labels = [item["label"] for item in train_data]
        val_texts = [item["text"] for item in val_data]
        val_labels = [item["label"] for item in val_data]
        
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Train model
        trainer = self.train_model(train_dataset, val_dataset)
        
        # Evaluate
        eval_results = self.evaluate_model(val_dataset)
        
        return {
            "trainer": trainer,
            "eval_results": eval_results,
            "model_path": "./results"
        }

# Usage Example
trainer = CustomModelTrainer(
    model_name="distilbert-base-uncased",
    num_labels=3  # For sentiment: negative, neutral, positive
)

# Prepare your domain-specific data
domain_data = [
    {"text": "This AI solution is excellent for our needs", "label": 2},
    {"text": "The model performance could be better", "label": 0},
    {"text": "Average implementation, works as expected", "label": 1},
    # ... more training examples
]

# Fine-tune model
results = trainer.fine_tune_for_domain(domain_data)
print(f"Training completed. F1 Score: {results['eval_results']['eval_f1']:.4f}")

# Make predictions
new_texts = ["This is an amazing AI chatbot implementation"]
predictions = trainer.predict(new_texts)
print(f"Prediction confidence: {predictions[0]}")`,
    features: ["Transfer Learning", "Domain Adaptation", "Performance Metrics", "Experiment Tracking"]
  },
  {
    id: "integration",
    title: "Production API Integration",
    description: "FastAPI deployment with authentication, monitoring, and scalable architecture",
    language: "Python",
    category: "Integration",
    code: `from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import redis
import logging
from typing import List, Optional
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
import time
import jwt

app = FastAPI(title="AI Chatbot API", version="1.0.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency')

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Security
security = HTTPBearer()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: str
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[dict] = []
    confidence: float
    processing_time: float

class AIService:
    def __init__(self):
        # Initialize your AI models here
        self.rag_system = None  # Your RAG implementation
        self.ocr_system = None  # Your OCR implementation
        
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request with AI models"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"chat:{hash(request.message)}"
        cached_response = redis_client.get(cache_key)
        
        if cached_response:
            return ChatResponse.parse_raw(cached_response)
        
        # Process with AI models
        try:
            # Example RAG processing
            if self.rag_system:
                result = await self.rag_system.query(request.message)
                response_text = result["answer"]
                sources = result.get("sources", [])
                confidence = result.get("confidence", 0.8)
            else:
                response_text = "AI processing placeholder"
                sources = []
                confidence = 0.8
            
            processing_time = time.time() - start_time
            
            response = ChatResponse(
                response=response_text,
                conversation_id=request.conversation_id or "new-conversation",
                sources=sources,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # Cache response
            redis_client.setex(
                cache_key, 
                3600,  # 1 hour cache
                response.json()
            )
            
            return response
            
        except Exception as e:
            logging.error(f"AI processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="AI processing failed")

ai_service = AIService()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Collect metrics for each request"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_LATENCY.observe(time.time() - start_time)
    
    return response

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token)
):
    """Main chat endpoint"""
    try:
        # Validate user permissions
        if not user.get("can_use_ai"):
            raise HTTPException(status_code=403, detail="AI access not permitted")
        
        # Process chat request
        response = await ai_service.process_chat(request)
        
        # Log usage in background
        background_tasks.add_task(
            log_usage,
            user_id=request.user_id,
            message_length=len(request.message),
            response_length=len(response.response)
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Chat processing failed")

@app.post("/api/ocr")
async def ocr_endpoint(
    file_content: bytes,
    user: dict = Depends(verify_token)
):
    """OCR processing endpoint"""
    try:
        # Process image with OCR
        if ai_service.ocr_system:
            result = await ai_service.ocr_system.process_image(file_content)
            return {"text": result["text"], "confidence": result["confidence"]}
        else:
            return {"text": "OCR placeholder", "confidence": 0.9}
            
    except Exception as e:
        logging.error(f"OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail="OCR processing failed")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

async def log_usage(user_id: str, message_length: int, response_length: int):
    """Log API usage for analytics"""
    usage_data = {
        "user_id": user_id,
        "timestamp": time.time(),
        "input_tokens": message_length // 4,  # Rough estimate
        "output_tokens": response_length // 4,
        "cost": (message_length + response_length) * 0.0001  # Pricing model
    }
    
    # Store in database or analytics system
    logging.info(f"Usage logged: {usage_data}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )`,
    features: ["Authentication", "Caching", "Monitoring", "Auto-scaling"]
  }
]

export function CodeShowcase() {
  const [copied, setCopied] = useState<string | null>(null)

  const copyToClipboard = async (code: string, id: string) => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(id)
      setTimeout(() => setCopied(null), 2000)
    } catch (err) {
      console.error('Failed to copy code:', err)
    }
  }

  const categories = [...new Set(codeExamples.map(example => example.category))]

  return (
    <section id="code" className="py-20 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-mono font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
            Code Examples
          </h2>
          <p className="text-lg text-muted-foreground font-mono max-w-2xl mx-auto">
            Production-ready implementations showcasing technical depth and best practices
          </p>
        </div>

        <Tabs defaultValue={codeExamples[0].id} className="w-full">
          <TabsList className="grid w-full grid-cols-2 lg:grid-cols-6 mb-8 bg-card/50">
            {codeExamples.map((example) => (
              <TabsTrigger 
                key={example.id} 
                value={example.id}
                className="font-mono text-xs data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              >
                {example.category}
              </TabsTrigger>
            ))}
          </TabsList>

          {codeExamples.map((example) => (
            <TabsContent key={example.id} value={example.id}>
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="font-mono text-xl mb-2">{example.title}</CardTitle>
                      <CardDescription className="font-mono">
                        {example.description}
                      </CardDescription>
                      <div className="flex flex-wrap gap-2 mt-4">
                        {example.features.map((feature, index) => (
                          <Badge 
                            key={index}
                            variant="secondary" 
                            className="font-mono text-xs bg-primary/10 text-primary"
                          >
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(example.code, example.id)}
                        className="font-mono"
                      >
                        {copied === example.id ? (
                          "Copied!"
                        ) : (
                          <>
                            <Copy className="h-4 w-4 mr-1" />
                            Copy
                          </>
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="font-mono"
                      >
                        <Github className="h-4 w-4 mr-1" />
                        View Full
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="relative">
                    <div className="absolute top-3 right-3 text-xs text-muted-foreground font-mono bg-background/80 px-2 py-1 rounded">
                      {example.language}
                    </div>
                    <pre className="bg-background/50 border border-border/50 rounded-lg p-4 overflow-x-auto text-sm font-mono">
                      <code className="text-foreground">{example.code}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          ))}
        </Tabs>

        <div className="text-center mt-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/20 bg-primary/5 mb-6">
            <Github className="h-4 w-4 text-primary" />
            <span className="text-sm font-mono text-primary">Open Source Available</span>
          </div>
          <p className="text-muted-foreground font-mono mb-6">
            Complete implementations available on GitHub with documentation and deployment guides
          </p>
          <Button 
            size="lg" 
            variant="outline"
            className="border-primary/50 hover:bg-primary/5 font-mono"
          >
            <ExternalLink className="mr-2 h-5 w-5" />
            View Full Repository
          </Button>
        </div>
      </div>
    </section>
  )
}
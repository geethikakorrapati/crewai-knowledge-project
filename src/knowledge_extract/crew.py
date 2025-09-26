# src/knowledge_extract/crew.py
import os, glob
from typing import List

from dotenv import load_dotenv
load_dotenv()  # make sure env is loaded first

from crewai import Agent, Crew, Process, LLM, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Add this import
import google.generativeai as genai


def _embedder_from_env():
    """
    Configure the Google GenAI SDK for REST and return the embedder config
    CrewAI/Chroma expects. We pass api_key/model_name (no unsupported kwargs).
    """
    provider = os.getenv("EMBEDDINGS_PROVIDER", "google-generativeai")
    model    = os.getenv("EMBEDDINGS_MODEL", "models/text-embedding-004")
    api_key  = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY found for embeddings.")

    # Force REST globally so Chroma's Google embedding function uses REST, not gRPC
    genai.configure(api_key=api_key, transport="rest")

    # Return only the supported fields for CrewAI → Chroma Google EF
    return {"provider": provider, "model_name": model, "api_key": api_key}


def _llm_from_env() -> LLM:
    model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
    return LLM(model=model, temperature=temperature, max_tokens=max_tokens)


def _pdf_knowledge_sources():
    kb_dir = os.path.join(os.getcwd(), "knowledge")
    abs_pdfs = sorted(glob.glob(os.path.join(kb_dir, "*.pdf")))
    if not abs_pdfs:
        raise FileNotFoundError(f"No PDFs found in {kb_dir}. Put your files there.")
    rel_pdfs = [os.path.basename(p) for p in abs_pdfs]
    return [PDFKnowledgeSource(file_paths=rel_pdfs, chunk_size=1200, chunk_overlap=150)]


@CrewBase
class KnowledgeExtract:
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config_path = os.path.join("src", "knowledge_extract", "config", "agents.yaml")
    tasks_config_path  = os.path.join("src", "knowledge_extract", "config", "tasks.yaml")

    shared_llm = _llm_from_env()

    @agent
    def qa_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_agent"],
            llm=self.shared_llm,
            verbose=True,
            allow_delegation=False,
        )
    @agent
    def validator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["validator_agent"],
            llm=self.shared_llm,
            verbose=True,
            allow_delegation=False,
        )
    @agent
    def formatter_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["formatter_agent"],
            llm=self.shared_llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def qa_task(self) -> Task:
        return Task(config=self.tasks_config["qa_task"],agent=self.qa_agent())
    
    @task
    def validation_task(self) -> Task:
        return Task(
            config=self.tasks_config["validation_task"],
            # The context is the output from the first task
            context=[self.qa_task()],
            agent=self.validator_agent()
        )
    @task
    def formatting_task(self) -> Task:
        return Task(
            config=self.tasks_config["formatting_task"],
            context=[self.validation_task()], # Takes context from the validation task
            agent=self.formatter_agent()
        )
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.qa_agent(), self.validator_agent(), self.formatter_agent()],
            tasks=[self.qa_task(), self.validation_task(), self.formatting_task()],
            process=Process.sequential,
            knowledge_sources=_pdf_knowledge_sources(),
            embedder=_embedder_from_env(),  # now REST is forced before Chroma embeds
            memory=False,
            verbose=True,
        )
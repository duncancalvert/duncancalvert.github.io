---
layout: post
title: The Roadmap to AI Technical Product Manager
date: 2025-05-18 16:04:10
description: A guide to AI upskilling for Technical Product Managers
tags: AI Machine_Learning Deep_Learning Research Neural_Networks Product_Management Agents
categories: data-science
typograms: true
---

## What Makes an AI Technical Product Manager Different Than a Normal TPM

The rise of AI and Agentic frameworks have redefined the landscape of product development and the role of a TPM. A traditional TPM focuses on building scalable systems, ensures alignment of engineering execution with product goals, and translates business needs into technical specs. In contrast, an AI TPM guides the development of probabilistic, data-dependent products where performance can vary across inputs, and success isn’t measured in "features shipped" but in model quality, inference efficiency, and real-world generalization.

AI systems introduce new kinds of architecture concerns: model training pipelines, feature stores, real-time inference latency, versioning of data/models, and monitoring for drift and degradation. The AI TPM must orchestrate this stack while aligning with model scientists, ML engineers, infra teams, data SMEs, end users, and compliance stakeholders. They are responsible not just for what gets built, but how learning systems are trained, deployed, scaled, and governed. AI TPMs live at the intersection of infrastructure, research, and imperfect .

---
## Skill Sets
<br />


### AI Product Skills
* <ins>AI product sense</ins>: understand what can, and importantly cannot, be solved by AI (i.e. AI is not a silver bullet, many processes and products are better served with non-AI solutions)
* <ins>AI experiment design</ins>: practice iterative hypothesis testing with quantitative evaluation. Lead with A/B test, user interviews, and user feedback loops wherever possible
* <ins>Market insight</ins>: build a deep understanding of the AI market, its competitive landscape, and emerging trends
* <ins>User Journeys</ins>: define clear user journeys that align to a strategic AI product philosophy and north star metric

### Traditional Data Science Expertise
* <ins>AI models</ins>: understand what the difference is between Random Forest, SVM, and KNN and when to use one over the other on a problem. 
* <ins>AI evaluation metrics</ins>: undestand the right metrics for each model and use case.
* Model Development Lifecycle (MDLC): 
* Machine Learning Operations (MLOps) processes and principles
* Python (OOP principles, Pandas, NumPy, Jupyter)
* Deep Learning (PyTorch/TensorFlow/Non-Transformer Neural Networks)
    * [(Class) Deep Learning Specialisation by Andrew Ng](https://www.coursera.org/specializations/deep-learning)
    * [(Book) Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)
* SQL
* Apache Spark/PySpark

### Gen AI Expertise
* Vector databases (Pinecone, Weaviate, Chroma)
* Model garden APIs (Azure, GCP, AWS, OpenAI)
* Models
    * Transformer model architecture
        * [(Book) Build a Large Language Model (From Scratch) by Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)
        * [(Video) Deep Dive into LLMs like ChatGPT by Andrej Karpathy](https://www.youtube.com/watch?v=7xTGNNLPyMI&ab_channel=AndrejKarpathy)
        * [(Paper) Attention is All You Need](https://arxiv.org/abs/1706.03762)
        * [(Class) Stanford CS229 - Machine Learning - Building Large Language Models (LLMs)](https://www.youtube.com/watch?v=9vM4p9NN0Ts&ab_channel=StanfordOnline)
    * Diffusion model architecture
    * GAN model architecture
* LLM benchmarks
    * [(Podcast Episode) AI Fundamentals: Benchmarks 101](https://www.latent.space/p/benchmarks-101)
    * [(Podcast Episode) Benchmarks 201: Why Leaderboards > Arenas >> LLM-as-Judge](https://www.latent.space/p/benchmarks-201)
* LLM evaluation strategies
* Prompt Engineering
    * [(Article) OpenAI Prompting Guide](https://platform.openai.com/docs/guides/text?api-mode=responses)
    * [(Website) Prompt Engineering Guide by DAIR.AI](https://www.promptingguide.ai/)
* LLM Observability (Langfuse and LangSmith)

### Agentic Expertise
* Agent Fundamentals
    * [(White Paper) Google Agents White Paper by Julia Wiesinger et al.](https://www.kaggle.com/whitepaper-agents)
    * [(White Paper) Google Agents Companion White Paper by Antonio Gulli et al.](https://www.kaggle.com/whitepaper-agent-companion)
    * [(Paper) ReAct: Synergizing Reasoning and Acting in Language Models by Shunyu Yao et al.](https://arxiv.org/abs/2210.03629)
* Agent Evals
    * [(Paper) Agent-as-a-Judge: Evaluate Agents with Agents](https://arxiv.org/abs/2410.10934)
* Agent Frameworks
    * LangChain
        * [(Class) AI Agents in LangGraph by DeepLearning.AI](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
    * LangGraph
    * LlamaIndex
        * [(Class) Building Agentic RAG with LlamaIndex by DeepLearning.AI](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)
    * Open AI Agent SDK
    * Mastra
* Agent Protocols
    * Anthropic Model Context Protocol (MCP)
        * [(Article) Why Every AI Builder Needs to Understand MCP](https://blog.neosage.io/p/why-every-ai-builder-needs-to-understand)
    * Google Agent-2-Agent (A2A)
* AI Integrated Development Environments (IDE): Cursor, Windsurf, or Replit
* Agentic Design Patterns
    * [(Article) Zero to One: Learning Agentic Patterns](https://www.philschmid.de/agentic-pattern)
    * [(Class) AI Agentic Design Patterns with AutoGen by DeepLearning.AI](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)
    * [(Class) Multi AI Agent Systems with crewAI by DeepLearning.AI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)

### General Technical Expertise
* Public cloud infrastructure (GCP, Azure, AWS)
* Data pipelines
    * Apache Airflow/GCP Composer
    * Dataflow
    * Apache Beam
    * Apache Kafka
* API and Backend skills
    * Develop backends with FastAPI or Flask 
    * Implement REST and streaming endpoints for AI services
    * Design authentication and rate limiting systems
    * Build WebSocket implementations for real-time AI interactions

### Soft Skills
* <ins>Stakeholder management</ins>: Adept at influencing executives and building consensus in a constantly changing and fast-paced environment. 
* <ins>Expert Storytelling</ins>: master product positioning and messaging. Create a proven track record of successfully positioning solutions, presentation, and public speaking for technology professionals and leaders
* <ins>Product Launch Experience</ins>: build up a knowledge of what to do at different stages of the product launch cycle, and how to do it
* <ins>Growth and Expansive Mindset</ins>: foster a curiosity to learn, growth mindset, and positive attitude (kind human policy)

### General Product Management
* Waterfall
* Agile (Scrum/Kanban)
* Continuous Integration/Continuous Delivery (CI/CD)
* DevOps and Site Reliability Engineering (SRE)
* Robust documentation
* FinOps


---

## Newsletters, Podcasts, and People
<br />

### Newsletters
* [The Sequence](https://thesequence.substack.com/): A weekly series that does technical deep dives on the latest AI/ML techniques 
* [The Batch @ DeepLearning.AI](https://www.deeplearning.ai/the-batch/): a weekly deep dive from Stanford Professor Andrew Ng
* [Data Points](https://www.deeplearning.ai/the-batch/tag/data-points/) 
* [Daily Zaps](https://www.dailyzaps.com/): high level tech news, not very technical
* [The MLOps Newsletter](https://mlops.substack.com/): technical with a specific focus on MLOps
* [Google Developer Program](https://developers.google.com/newsletter): stay up to date with the latest GCP releases
* [The Variable](https://medium.com/towards-data-science/newsletter): a curated list of articles/tutorials from Towards Data Science, the data science channel in Medium
* [The Download from MIT Technology Review](https://www.technologyreview.com/topic/download-newsletter/): a higher level tech news roundup
* [Turing Post](https://www.turingpost.com/) 

### Podcasts
* [Practical AI by Changelog](https://podcasts.apple.com/us/podcast/practical-ai/id1406537385)
* [Inference by Turing Post](https://www.youtube.com/playlist?list=PLRRoCwK1ZTNCAZXXOswpIYQqzMgT4swsI)
* [Latent Space: The AI Engineer Podcast](https://www.latent.space/podcast)

### People to Follow
* [Yann LeCun](https://www.linkedin.com/in/yann-lecun/)
* Andrej Karpathy
* Fei-Fei Li
* [EugeneYan](https://eugeneyan.com/subscribe): ML, RecSys, LLMs, and engineering


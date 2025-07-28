---
layout: post
title: The Roadmap to AI Technical Product Manager
date: 2025-05-18 16:04:10
description: A guide to AI upskilling for Technical Product Managers
tags: AI Machine_Learning Deep_Learning Research Neural_Networks Product_Management Agents
categories: data-science
typograms: true
---

<br>
<br>
<p style="text-align: center;">
    <em>"Be stubborn on vision but flexible on details."</em><br>
    — Jeff Bezos 
</p>
<br>
<br>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
      {% include figure.liquid loading="eager" path="assets/img/posts_the_roadmap_to_ai_pm/Map_Denise Jans Unsplash.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
    Source: Photo by Denise Jans on Unsplash
</div>
<br>
<br>

## Table of Contents

---

<!-- TOC -->

- [Table of Contents](#table-of-contents)
- [What Makes an AI Technical Product Manager Different](#what-makes-an-ai-technical-product-manager-different)
- [Skill Sets](#skill-sets)
    - [Traditional Data Science](#traditional-data-science)
    - [Gen AI & Foundation Models](#gen-ai--foundation-models)
    - [Retrieval Augmented Generation](#retrieval-augmented-generation)
    - [Agentic AI](#agentic-ai)
    - [General Technical Skills](#general-technical-skills)
    - [AI Product Skills](#ai-product-skills)
    - [General Product Management](#general-product-management)
- [Newsletters, Podcasts, and People](#newsletters-podcasts-and-people)
    - [Newsletters](#newsletters)
    - [Podcasts](#podcasts)
    - [People to Follow](#people-to-follow)

<!-- /TOC -->
<br>

## What Makes an AI Technical Product Manager Different

The rapid rise of AI and Agentic frameworks has redefined the landscape of product development and the role of a TPM. A traditional TPM focuses on building scalable systems, ensures alignment of engineering with product, and translates business needs into technical specs. They primarily build with tools that are time-tested, relatively stable, and well documented.

In contrast, AI TPMs live at the intersection of infrastructure, cutting edge research, and business outcomes. They guide the development of probabilistic, data-dependent products where performance varies widely across inputs, and success isn’t measured in "features shipped" but in hard to measure metrics like model quality, inference efficiency, and real-world generalization. Add to that the fact that tools are shifting under their feet with vendors and the open-source community launching new frameworks based on the latest cutting edge agentic and Gen AI research.

Instead of the generally linear Software Development Lifecycle (SDLC) used to build traditional products, building AI systems require a highly iterative Model Development Lifecycle (MDLC). This involves continuously tweaking model training pipelines, feature stores, real-time inference latency, versioning of data/models, and monitoring for drift and degradation. This must be done in alignment with a motely crew of data scientists, ML engineers, infra teams, data SMEs, end users, and model governance stakeholders.

<br>

## Skill Sets

---

<!------------------ Section --------------------->

### Traditional Data Science

<details>
  <summary><b>Probability & Statistics</b></summary>
  <ul>
    <li><a href="https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/">(Book) Practical Statistics for Data Scientists by Peter Bruce</a></li>
  </ul>
</details>

<details>
  <summary><b>Machine Learning Theory</b></summary>
  <ul>
    <li>Supervised vs. Unsupervised Learning</li>
    <li>Regression, Classification, Clustering, Association, and Dimensionality Reduction</li>
    <li>Understand the different models (Random Forest, SVM, K-means, etc.) and when to use one over the other</li>
    <li>Hyperparameters and issues with each model.</li>
    <li>Grid vs. Random Search (Gradient Descent)</li>
  </ul>
</details>

<details>
  <summary><b>Deep Learning Theory</b></summary>
  <ul>
    <li>Model architectures for Artificial Neural Networks (ANN) and deep learning models (DNN, CNN, RNN, etc.).</li>
      <ul>  
        <li>Activation functions</li>
        <li>Loss functions/cost functions</li>
        <li>Backpropagation</li>
        <li>Model weights and biases</li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>AI Evaluation Metrics</b></summary>
  <ul>
    <li>Build an intuitive understanding of the right metrics for each model and use case.</li>
    <li>Recognize areas of concern or blind spots for each metric.</li>
  </ul>
</details>

<details>
  <summary><b>ML & Deep Learning Frameworks</b></summary>
    <ul>
      <li>Scikit-Learn</li>
      <li>PyTorch</li>
      <li>TensorFlow</li>
      <li>JAX</li>
      <li>Resources</li>
        <ul>
          <li><a href="https://www.coursera.org/specializations/deep-learning">(Class) Deep Learning Specialisation by Andrew Ng</a></li>
          <li><a href="https://www.deeplearningbook.org/">(Book) Deep Learning by Ian Goodfellow</a></li>
          <li><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/">(Book) Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition by Aurélien Géron</a></li>
          <li><a href="https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ">(Videos) Neural Networks: Zero to Hero by Andrej Karpathy</a></li>
        </ul>
    </ul>
</details>

<details>
  <summary><b>Model Development Lifecycle (MDLC)</b></summary>
  <ul>
    <li>Understand the end-to-end process of building, testing, deploying, and monitoring machine learning models.</li>
  </ul>
</details>

<details>
  <summary><b>Machine Learning Operations (MLOps)</b></summary>
  <ul>
    <li>Learn the principles and practices of maintaining and scaling ML workflows in production environments.</li>
    <li>General Resources</li>
      <ul>
        <li><a href="https://a.co/d/eIRcpD7">(Book) Designing Machine Learning Systems by Chip Huyen</a></li>
        <li><a href="https://github.com/GokuMohandas/Made-With-ML">(Class) Made With ML</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Python</b></summary>
  <ul>
    <li>Learn Object-oriented programming (OOP) principles.</li>
    <li>Proficiency in Pandas and NumPy for data manipulation.</li>
    <li>Use Jupyter notebooks for exploration and experimentation.</li>
  </ul>
</details>

<details>
  <summary><b>SQL</b></summary>
  <ul>
    <li>Ensure fluency in querying and manipulating structured data from relational databases.</li>
  </ul>
</details>

<details>
  <summary><b>Apache Spark / PySpark</b></summary>
  <ul>
    <li>Leverage distributed computing for large-scale data processing.</li>
    <li>Use PySpark for writing scalable, Python-based ETL and analysis pipelines.</li>
  </ul>
</details>

<details>
  <summary><b>Data Warehouse / Data Lakehouse</b></summary>
  <ul>
    <li><a href="https://www.databricks.com/" target="_blank" rel="noopener noreferrer">Databricks</a></li>
    <li><a href="https://www.snowflake.com/" target="_blank" rel="noopener noreferrer">Snowflake</a></li>
    <li><a href="https://cloud.google.com/bigquery" target="_blank" rel="noopener noreferrer">GCP BigQuery</a></li>
  </ul>
</details>

---

<!------------------ Section --------------------->

### Gen AI & Foundation Models

<details>
  <summary><b>Cloud Model APIs</b></summary>
  <ul>
    <li><a href="https://azure.microsoft.com/en-us/products/ai-model-catalog">Azure - AI Foundry</a></li>
    <li><a href="https://cloud.google.com/model-garden">GCP - Vertex AI Model Garden</a></li>
    <li><a href="https://aws.amazon.com/bedrock/">AWS - Amazon Bedrock</a></li>
    <li><a href="https://openai.com/api/">OpenAI</a></li>
  </ul>
</details>

<details>
  <summary><b>Transformer Theory</b></summary>
  <ul>
    <li>Architecture & Training</li>
      <ul>
        <li>Attention Mechanism</li>
        <li>Positional Encoding</li>
        <li>Tokenization & Embeddings</li>
        <li>Decoder-Only, Encoder-Only, and Encoder-Decoder Models</li>
        <li>Hyperparameter Tuning</li>
          <ul>
            <li>Temperature</li>
            <li>top-K</li>
            <li>top-P</li>
          </ul>
      </ul>
    <li>Resources:</li>
    <ul>
        <li><a href="https://www.manning.com/books/build-a-large-language-model-from-scratch">(Book) Build a Large Language Model (From Scratch) by Sebastian Raschka</a></li>
        <li><a href="https://a.co/d/3POHRks">(Book) Hands-On Large Language Models by Jay Alammar</a></li>
        <li><a href="https://a.co/d/fI0MdEB">(Book) Natural Language Processing with Transformers by Lewis Tunstall</a></li>
        <li><a href="https://www.youtube.com/watch?v=7xTGNNLPyMI&ab_channel=AndrejKarpathy">(Video) Deep Dive into LLMs like ChatGPT by Andrej Karpathy</a></li>
        <li><a href="https://arxiv.org/abs/1706.03762">(Paper) Attention is All You Need</a></li>
        <li><a href="https://www.youtube.com/watch?v=9vM4p9NN0Ts&ab_channel=StanfordOnline">(Class) Stanford CS229 - Machine Learning - Building Large Language Models (LLMs)</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Diffusion Models</b></summary>
  <ul>
    <li></li>
  </ul>
</details>

<details>
  <summary><b>Generative Adversarial Networks (GAN)</b></summary>
  <ul>
    <li><a href="https://www.deeplearning.ai/courses/generative-adversarial-networks-gans-specialization/" target="_blank" rel="noopener noreferrer">(Course) DeepLearning.AI - GAN Specialization</a></li>
    <li><a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noopener noreferrer">(Paper) Original GAN Paper (Goodfellow et al.)</a></li>
    <li><a href="https://www.tensorflow.org/tutorials/generative/dcgan" target="_blank" rel="noopener noreferrer">(Tutorial) TensorFlow Deep Convolutional GAN Tutorial</a></li>
    <li><a href="https://www.youtube.com/watch?v=8L11aMN5KY8" target="_blank" rel="noopener noreferrer">(Video) A Friendly Introduction to GANs by Serrano Academy</a></li>
  </ul>
</details>

<details>
  <summary><b>LLM Fine-Tuning</b></summary>
  <ul>
    <li>Compute Efficiency Techniques</li>
      <ul>
        <li>LoRA</li>
        <li>QLoRA</li>
        <li>PEFT</li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Benchmarks</b></summary>
  <ul>
    <li><a href="https://www.latent.space/p/benchmarks-101">(Podcast) AI Fundamentals: Benchmarks 101</a></li>
    <li><a href="https://www.latent.space/p/benchmarks-201">(Podcast) Benchmarks 201: Why Leaderboards > Arenas >> LLM-as-Judge</a></li>
  </ul>
</details>

<details>
  <summary><b>LLM Evaluation</b></summary>
  <ul>
    <li>LLM Eval Metrics</li>
      <ul>
        <li>Statistical Metrics</li>
          <ul>
            <li><a href="https://en.wikipedia.org/wiki/BLEU">BLEU</a></li>
            <li><a href="https://en.wikipedia.org/wiki/ROUGE_(metric)">ROUGE (Recall-Oriented Understudy for Gisting Evaluation)</a></li>
            <li><a href="https://en.wikipedia.org/wiki/METEOR">METEOR (Metric for Evaluation of Translation with Explicit Ordering)</a></li>
            <li><a href="https://en.wikipedia.org/wiki/Levenshtein_distance">Levenshtein Distance</a></li>
          </ul>
        <li>LLM-as-Judge Metrics</li>
      </ul>
    <li>LLM Eval Tools</li>
      <ul>
        <li><a href="https://docs.ragas.io">Ragas</a></li>
      </ul>
    <li>LLM-as-Judge Techniques</li>
      <ul>
        <li>Pairwise Comparison</li>
        <li>Evaluation by Criteria (Reference Free)</li>
        <li>Evaluation by Criteria (Reference-Based)</li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Observability</b></summary>
  <ul>
    <li>Langfuse</li>
    <li><a href="https://www.langchain.com/langsmith">LangSmith</a>: a developer platform for inspecting, tracing, and evaluating LLM-powered applications built with LangChain or other orchestration frameworks. It enables fine-grained logging of prompts, model inputs/outputs, tool invocations, and intermediate steps, while supporting automated and manual evaluation workflows for performance, latency, and correctness.</li>
  </ul>
</details>

<details>
  <summary><b>LLM Interpretability</b></summary>
  <ul>
    <li>Anthropic's Interpretability Team</li>
      <ul>
        <li><a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html">Dictionary Learning</a></li>
        <li><a href="https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning">Monosemanticity</a></li>
        <li><a href="https://transformer-circuits.pub/2025/attribution-graphs/biology.html">Attributional Graphs</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Safety & Security</b></summary>
  <ul>
    <li>Content Filtering</li>
      <ul>
        <li><a href="https://cloud.google.com/security-command-center/docs/model-armor-overview">GCP Model Armor</a>
        </li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Prompt/Context Engineering</b></summary>
  <ul>
    <li><a href="https://platform.openai.com/docs/guides/text?api-mode=responses">(Article) OpenAI Prompting Guide</a></li>
    <li><a href="https://www.promptingguide.ai/">(Website) Prompt Engineering Guide by DAIR.AI</a></li>
  </ul>
</details>

<details>
  <summary><b>AI Engineering</b></summary>
  <ul>
    <li><a href="https://a.co/d/gr8eiN1">(Book) AI Engineering: Building Applications with Foundation Models by Chip Huyen</a></li>
  </ul>
</details>

---

<!------------------ Section --------------------->

### Retrieval Augmented Generation

<details>
  <summary><b>RAG Fundamentals</b></summary>
  <ul>
    <li>Vector embeddings</li>
    <li>Chunking</li>
    <li>Hybrid retrieval</li>
    <li>General Resources</li>
      <ul>
        <li>
          <a href="https://github.com/NirDiamant/RAG_Techniques">(GitHub) RAG Techniques by Nir Diamant</a>
        </li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>RAG Evaluation</b></summary>
  <ul>
    <li>RAG Eval Metrics</li>
    <ul>
      <li>Context Precision</li>
      <li>Context Recall</li>
      <li>Content Entities Recall</li>
      <li>Noise Sensitivity</li>
      <li>Response Relevance</li>
      <li>Faithfulness</li>
      <li>Multimodal Faithfulness</li>
      <li>Multimodal Relevance</li>
    </ul>
  <li>RAG Eval Tools</li>
    <ul>
      <li>Ragas</li>
    </ul>
  </ul>
</details>

<details>
  <summary><b>Vector Databases</b></summary>
  <ul>
    <li>Vector Search Prototyping Libraries</li>
      <ul>
        <li><a href="https://faiss.ai/">FAISS</a></li>
        <li><a href="https://github.com/nmslib/hnswlib">HNSWlib</a></li>
      </ul>
    <li>Production Databases</li>
      <ul>
        <li>Pinecone</li>
        <li>Weaviate</li>
        <li>Chroma</li>
        <li>Elasticsearch</li>
        <li>Milvus</li>
      </ul>
  </ul>
</details>

---

<!------------------ Section --------------------->

### Agentic AI

<details>
  <summary><b>Agent Fundamentals</b></summary>
  <ul>
    <li><a href="https://www.kaggle.com/whitepaper-agents">(White Paper) Google Agents White Paper by Julia Wiesinger et al.</a></li>
    <li><a href="https://www.kaggle.com/whitepaper-agent-companion">(White Paper) Google Agents Companion by Antonio Gulli et al.</a></li>
    <li><a href="https://arxiv.org/abs/2505.10468">(Paper) AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges</a></li>
    <li><a href="https://arxiv.org/abs/2210.03629">(Paper) ReAct: Synergizing Reasoning and Acting in Language Models by Shunyu Yao et al.</a></li>
    <li><a href="https://huggingface.co/learn/agents-course/en/unit0/introduction">(Course) HuggingFace AI Agents Course</a></li>
  </ul>
</details>

<details>
  <summary><b>Agent Evals</b></summary>
  <ul>
    <li><a href="https://arxiv.org/abs/2410.10934">(Paper) Agent-as-a-Judge: Evaluate Agents with Agents</a></li>
  </ul>
</details>

<details>
  <summary><b>Agent Frameworks</b></summary>
  <ul>
    <li>LangChain</li>
    <li>LangGraph
      <ul>
        <li><a href="https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/">(Class) AI Agents in LangGraph by DeepLearning.AI</a></li>
      </ul>
    </li>
    <li>LlamaIndex
      <ul>
        <li><a href="https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/">(Class) Building Agentic RAG with LlamaIndex</a></li>
      </ul>
    </li>
    <li>OpenAI Agent SDK</li>
    <li>Mastra</li>
  </ul>
</details>

<details>
  <summary><b>Agent Protocols</b></summary>
  <ul>
    <li>Anthropic Model Context Protocol (MCP)
      <ul>
        <li><a href="https://blog.neosage.io/p/why-every-ai-builder-needs-to-understand">(Article) Why Every AI Builder Needs to Understand MCP</a></li>
      </ul>
    </li>
    <li>Google Agent-2-Agent (A2A)</li>
  </ul>
</details>

<details>
  <summary><b>AI Integrated Development Environments (IDEs)</b></summary>
  <ul>
    <li>Cursor</li>
    <li>Windsurf</li>
    <li>Replit</li>
  </ul>
</details>

<details>
  <summary><b>Agentic Design Patterns</b></summary>
  <ul>
    <li><a href="https://www.philschmid.de/agentic-pattern">(Article) Zero to One: Learning Agentic Patterns</a></li>
    <li><a href="https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/">(Class) AI Agentic Design Patterns with AutoGen</a></li>
    <li><a href="https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/">(Class) Multi AI Agent Systems with crewAI</a></li>
  </ul>
</details>

---

<!------------------ Section --------------------->

### General Technical Skills

<details>
  <summary><b>Public Cloud Infrastructure</b></summary>
  <ul>
    <li><a href="https://cloud.google.com/">Google Cloud Platform (GCP)</a></li>
    <li><a href="https://azure.microsoft.com/en-us/">Microsoft Azure</a></li>
    <li><a href="https://aws.amazon.com/">Amazon Web Services (AWS)</a></li>
  </ul>
</details>

<details>
  <summary><b>Data Pipelines</b></summary>
  <ul>
    <li>Apache Airflow</li>
      <ul>
        <li>GCP Composer</li>
        <li>Amazon Managed Workflows for Apache Airflow (MWAA)</li>
        <li>Azure Workflow Orchestration Manager</li>
      </ul>
    <li>Dataflow</li>
    <li>Apache Beam</li>
    <li>Apache Kafka</li>
  </ul>
</details>

<details>
  <summary><b>API and Backend Skills</b></summary>
  <ul>
    <li>Develop backends with FastAPI or Flask</li>
    <li>Implement REST and streaming endpoints for AI services</li>
    <li>Design authentication and rate-limiting systems</li>
    <li>Build WebSocket implementations for real-time AI interactions</li>
  </ul>
</details>

---

<!------------------ Section --------------------->

### AI Product Skills

<details>
  <summary><b>AI Product Sense</b></summary>
  <ul>
    <li>Understand what can, and importantly cannot, be solved by AI (i.e. AI is not a silver bullet, many processes and products are better served with non-AI solutions)</li>
  </ul>
</details>

<details>
  <summary><b>AI Experiment Design</b></summary>
  <ul>
    <li>Practice iterative hypothesis testing with quantitative evaluation. </li>
    <li>Lead with A/B test, user interviews, and user feedback loops where possible</li>
  </ul>
</details>

<details>
  <summary><b>Market Insight</b></summary>
  <ul>
    <li>Build a deep understanding of the AI market, its competitive landscape, and emerging trends</li>
  </ul>
</details>

<details>
  <summary><b>User Journeys</b></summary>
  <ul>
    <li>Define clear user journeys aligned with a strategic AI product philosophy and a north star metric.</li>
  </ul>
</details>

---

<!------------------ Section --------------------->

### General Product Management

<details>
  <summary><b>Project Management Frameworks</b></summary>
  <ul>
    <li><ins>Waterfall</ins>: A traditional, sequential approach where each project phase is completed before the next begins. Each phase has specific deliverables and a review process, making it suitable for projects with clearly defined requirements and predictable outcomes. However, it offers limited flexibility for changes once a phase is complete.</li>
    <li><ins>Agile</ins>: An iterative and incremental approach, suitable for projects with evolving requirements</li>
      <ul>
        <li><ins>Scrum</ins>: structured roles, sprints, and ceremonies.</li>
        <li><ins>Kanban</ins>: visual flow-based system emphasizing WIP limits and continuous delivery.</li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Continuous Integration / Continuous Delivery (CI/CD)</b></summary>
  <ul>
    <li>Automate testing, building, and deployment to speed up release cycles and improve reliability.</li>
  </ul>
</details>

<details>
  <summary><b>DevOps and Site Reliability Engineering (SRE)</b></summary>
  <ul>
    <li>Bridge development and operations to ensure scalable, stable, and reliable systems.</li>
    <li>SRE focuses on uptime, latency, monitoring, and incident response with a software engineering mindset.</li>
  </ul>
</details>

<details>
  <summary><b>Robust Documentation</b></summary>
  <ul>
    <li>Ensure product documentation is clear, current, and accessible to cross-functional teams.</li>
  </ul>
</details>

<details>
  <summary><b>FinOps</b></summary>
  <ul>
    <li>Manage cloud financial operations to maximize efficiency and optimize cost.</li>
  </ul>
</details>

<details>
  <summary><b>Stakeholder Management</b></summary>
  <ul>
    <li>Adept at influencing executives and building consensus in a constantly changing and fast-paced environment.</li>
  </ul>
</details>

<details>
  <summary><b>Expert Storytelling</b></summary>
  <ul>
    <li>Craft compelling product messaging and present effectively to diverse audiences.</li>
  </ul>
</details>

<details>
  <summary><b>Product Launch Experience</b></summary>
  <ul>
    <li>Know what to do at each product launch stage and how to execute effectively to get things over the finish line</li>
  </ul>
</details>

<details>
  <summary><b>Growth and Expansive Mindset</b></summary>
  <ul>
    <li>Foster a curiosity to learn, a growth mindset, a positive attitude, and a "kind human" policy.</li>
  </ul>
</details>

<br>

## Newsletters, Podcasts, and People

---

### Newsletters

- [The Sequence](https://thesequence.substack.com/): A weekly series that does technical deep dives on the latest AI/ML techniques
- [The Batch @ DeepLearning.AI](https://www.deeplearning.ai/the-batch/): a weekly deep dive from Stanford Professor Andrew Ng
- [Data Points](https://www.deeplearning.ai/the-batch/tag/data-points/)
- [Daily Zaps](https://www.dailyzaps.com/): high level tech news, not very technical
- [The MLOps Newsletter](https://mlops.substack.com/): technical with a specific focus on MLOps
- [Google Developer Program](https://developers.google.com/newsletter): stay up to date with the latest GCP releases
- [The Variable](https://medium.com/towards-data-science/newsletter): a curated list of articles/tutorials from Towards Data Science, the data science channel in Medium
- [The Download from MIT Technology Review](https://www.technologyreview.com/topic/download-newsletter/): a higher level tech news roundup
- [Turing Post](https://www.turingpost.com/subscribe?ref=WAGU23hEVa)
- [SwirlAI](https://www.newsletter.swirlai.com/): MLOps and data engineering focused newsletter with great visualizations

### Podcasts

- [Practical AI by Changelog](https://podcasts.apple.com/us/podcast/practical-ai/id1406537385)
- [Inference by Turing Post](https://www.youtube.com/playlist?list=PLRRoCwK1ZTNCAZXXOswpIYQqzMgT4swsI)
- [Latent Space: The AI Engineer Podcast](https://www.latent.space/podcast)

### People to Follow

- [Yann LeCun](https://www.linkedin.com/in/yann-lecun/): Chief AI Scientist at Meta and a pioneer in optical character recognition (OCR) and convolutional neural networks (CNN). A Turing Award winner and one of the three "Godfathers of AI".
- [Andrej Karpathy](https://karpathy.ai/): Former director of Autopilot at Tesla, co-founder of OpenAI, and prolific AI educator.
- [Fei-Fei Li](https://www.linkedin.com/in/fei-fei-li-4541247/): Stanford CS professor, co-director of the Stanford Institute for Human-Centered AI, inventor of ImageNet, and former Chief Scientist of AI/ML at GCP.
- [Eugene Yan](https://eugeneyan.com/subscribe): ML, RecSys, LLMs, and engineering
- [Andrew Ng](https://www.andrewng.org/): founder of Coursera, DeepLearning.AI, Stanford AI computer science professor, and neural network pioneer

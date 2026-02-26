---
layout: post
title: The Roadmap to AI Product Manager
date: 2025-05-18 16:04:10
description: A guide to the AI product management landscape
tags: AI Machine_Learning Deep_Learning Research Neural_Networks Product_Management Agents
categories: data-science
toc:
  beginning: false
---

<br>

<p style="text-align: center;">
    <em>"Be stubborn on vision but flexible on details."</em><br>
    — Jeff Bezos 
</p>

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


## What Makes an AI Product Manager
---
I frequently get asked by product managers, students, and engineers I work with, "how can I upskill on AI and especially AI product management". The AI landscape is moving at breakneck speed with new models, frameworks, and capabilities emerging weekly, making it challenging to know where to start or what to prioritize. This guide provides a structured  path through the essential skills needed to excel as an AI PM, organized by domain and prioritized for practical application.

So how does an AI Product Manager differ from a traditional PM? While traditional PMs focus on building scalable systems with time-tested, stable tools, AI PMs operate in a fundamentally different environment. They work at the intersection of infrastructure, cutting-edge research, and business outcomes, guiding the development of probabilistic, data-dependent products where performance varies across inputs. Success isn't measured in "features shipped" but in nuanced metrics like model quality, inference efficiency, and real-world generalization. The tools themselves are constantly shifting, with vendors and the open-source community launching new frameworks based on the latest agentic and Gen AI research.

The development process also differs dramatically. Instead of the generally linear Software Development Lifecycle (SDLC), AI systems require a a blend of linear SDLC and highly iterative Model Development Lifecycle (MDLC). This involves continuously tweaking model training pipelines, feature stores, real-time inference latency, versioning of data and models, and monitoring for drift and degradation—all while coordinating with a diverse team of data scientists, ML engineers, infrastructure teams, data SMEs, end users, and model governance stakeholders. This guide maps out the skills you'll need to navigate this complex, fast-moving landscape successfully.

<br>

## Skill Sets

<!------------------ Section --------------------->

### Traditional Data Science
---
A strong foundation in traditional data science is essential for AI PMs because it provides the fundamental language and concepts needed to communicate effectively with data scientists and ML engineers. Understanding probability, statistics, machine learning theory, and evaluation metrics enables PMs to make informed decisions about model selection, set realistic performance expectations, and translate technical capabilities into business value. Without this knowledge, PMs risk making promises that can't be delivered, misinterpreting model performance, or failing to identify when simpler statistical or deterministic methods might be more appropriate than complex ML or Gen AI solutions.

<details>
  <summary><b>Probability & Statistics</b></summary>
  <ul>
    <li>Fundamental Concepts</li>
      <ul>
        <li>Probability distributions (normal, binomial, Poisson, exponential)</li>
        <li>Conditional probability and Bayes' theorem</li>
        <li>Expected value, variance, and standard deviation</li>
        <li>Central Limit Theorem</li>
        <li>Law of Large Numbers</li>
      </ul>
    <li>Statistical Inference</li>
      <ul>
        <li>Hypothesis testing (null hypothesis, p-values, significance levels)</li>
        <li>Confidence intervals</li>
        <li>Type I and Type II errors</li>
        <li>Statistical power</li>
        <li>Bayesian vs. frequentist approaches</li>
      </ul>
    <li>Descriptive Statistics</li>
      <ul>
        <li>Measures of central tendency (mean, median, mode)</li>
        <li>Measures of dispersion (range, variance, standard deviation, IQR)</li>
        <li>Skewness and kurtosis</li>
        <li>Percentiles and quartiles</li>
      </ul>
    <li>Correlation and Causation</li>
      <ul>
        <li>Understanding correlation coefficients (Pearson, Spearman)</li>
        <li>Distinguishing correlation from causation</li>
        <li>Confounding variables</li>
        <li>Spurious correlations</li>
      </ul>
    <li>Sampling Methods</li>
      <ul>
        <li>Random sampling</li>
        <li>Stratified sampling</li>
        <li>Cluster sampling</li>
        <li>Bias in sampling</li>
        <li>Sample size determination</li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Machine Learning Theory</b></summary>
  <ul>
    <li>Supervised vs. unsupervised learning</li>
    <li>Regression, classification, clustering, association, and dimensionality reduction</li>
    <li>Familiarity with statistics and ML models and when to use one over the other</li>
      <ul>
          <li>Linear Regression</li>
          <li>Logistic Regression</li>
          <li>Random Forest</li>
          <li>Support Vector Machines (SVM)</li>
          <li>eXtreme Gradient Boosting (XGBoost)</li>
          <li>K-means Clustering</li>
          <li>Light Gradient Boosting Machine (LightGBM)</li>
          <li>Categorical Boosting (CatBoost)</li>
      </ul>
    <li>Model hyperparameters</li>
    <li>Grid vs. random search (gradient descent)</li>
    <li>Loss functions</li>
    <li>Overfitting and underfitting</li>
    <li>Bias and variance</li>
    <li>Model evaluation techniques (train/test split, cross validation, metrics)</li>
  </ul>
</details>

<details>
  <summary><b>Deep Learning Theory</b></summary>
    <ul>
      <li>Model architectures for Artificial Neural Networks (ANN) and deep learning models (DNN, CNN, RNN, etc.)</li>
        <ul>
          <li>Activation functions</li>
            <ul>
              <li>Rectified Linear Unit (ReLU)</li>
                <ul>
                  <li>Leaky ReLU</li>
                  <li>Softplus/SmoothReLU</li>
                  <li>Parametric ReLU (PReLU)</li>
                  <li>Exponential Linear Unit (ELU)</li>
                  <li>Gaussian Error Linear Unit (GELU)</li>
                </ul>
              <li>Sigmoid</li>
            </ul>
          <li>Loss functions/cost functions</li>
            <ul>
              <li>SME</li>
            </ul>
          <li>Optimizers</li>
            <ul>
              <li>Adadelta</li>
              <li>Adafactor</li>
              <li>Adaptive Gradient Algorithm (AdaGrad)</li>
              <li>Adaptive Moment Estimation (Adam)</li>
              <li>Adam with Weight Decay (AdamW)</li>
              <li>Adamax</li>
              <li>Follow-the-Regularized-Leader (Ftrl)</li>
              <li>Lion</li>
              <li>LossScaleOptimizer</li>
              <li>Nesterov-accelerated Adaptive Moment Estimation (Nadam)</li>
              <li>Root Mean Square Propagation (RMSprop)</li>
              <li>Stochastic Gradient Descent (SGD)</li>
            </ul>
          <li>Backpropagation</li>
          <li>Epochs</li>
          <li>Learning Rate</li>
          <li>Batch Size</li>
          <li>Regularization</li>
            <ul>
              <li>Early Stopping</li>
              <li>Parameter Norm Penalties
                <ul>
                  <li>L1 Regularization</li>
                  <li>L2 Regularization</li>
                  <li>Max-norm Regularization</li>
                </ul>
              </li>
              <li>Dataset Augmentation</li>
              <li>Noise Robustness</li>
              <li>Sparse Representations</li>
            </ul>
          <li>Model weights and biases</li>
          <li><ins>Embedding dimensionality</ins>: Lower dimensions can lead to less accuracy, more lossy compression. Higher can lead to overfitting and slow training time. A good starting point for the number of embedding dimensions is Dimensions \(\approx \sqrt[4]{\mathrm{Possible\ values}}\)</li>
        </ul>
    </ul>
</details>

<details>
  <summary><b>AI Evaluation Metrics</b></summary>
  <ul>
    <li>Theory</li>
      <ul>
        <li>Build an intuitive understanding of the right metrics for each model and use case, especially trade-offs.</li>
        <li>Recognize areas of concern or blind spots for each metric.</li>
        <li>Consider business context when selecting evaluation metrics (e.g., cost of false positives vs. false negatives).</li>
      </ul>
    <li>Regression Eval Metrics & Techniques</li>
      <ul>
        <li>Mean Absolute Error (MAE)</li>
        <li>Mean Squared Error (MSE)</li>
        <li>Root Mean Squared Error (RMSE)</li>
        <li>Mean Absolute Percentage Error (MAPE)</li>
        <li>R-squared (R²) / Coefficient of Determination</li>
        <li>Adjusted R-squared</li>
        <li>Mean Bias Error (MBE)</li>
      </ul>
    <li>Classification Eval Metrics & Techniques</li>
      <ul>
        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall (Sensitivity)</li>
        <li>F1-Score (harmonic mean of precision and recall)</li>
        <li>F-beta Score (weighted F1 for precision/recall trade-offs)</li>
        <li>Specificity (True Negative Rate)</li>
        <li>Confusion Matrix</li>
        <li>Precision-Recall Curve</li>
        <li>Receiver Operating Characteristic (ROC) Curve</li>
        <li>Area Under the ROC Curve (AUC-ROC)</li>
        <li>Area Under the Precision-Recall Curve (AUC-PR)</li>
        <li>Log Loss (Logarithmic Loss)</li>
        <li>Matthews Correlation Coefficient (MCC)</li>
      </ul>
    <li>Multi-class Classification Metrics</li>
      <ul>
        <li>Macro-averaged metrics (precision, recall, F1)</li>
        <li>Micro-averaged metrics</li>
        <li>Weighted-averaged metrics</li>
        <li>Per-class metrics</li>
      </ul>
    <li>Imbalanced Dataset Metrics</li>
      <ul>
        <li>When to use precision vs. recall vs. F1-score</li>
        <li>Precision-Recall AUC (often better than ROC-AUC for imbalanced data)</li>
        <li>Balanced Accuracy</li>
        <li>Cohen's Kappa</li>
      </ul>
    <li>Clustering Metrics</li>
      <ul>
        <li>Silhouette Score</li>
        <li>Davies-Bouldin Index</li>
        <li>Calinski-Harabasz Index</li>
        <li>Inertia (within-cluster sum of squares)</li>
        <li>Adjusted Rand Index (ARI) - for labeled data</li>
        <li>Normalized Mutual Information (NMI) - for labeled data</li>
      </ul>
    <li>Evaluation Techniques</li>
      <ul>
        <li>Train/Test Split</li>
        <li>K-Fold Cross-Validation</li>
        <li>Stratified K-Fold Cross-Validation</li>
        <li>Time Series Cross-Validation (for temporal data)</li>
        <li>Bootstrap Sampling</li>
        <li>Leave-One-Out Cross-Validation (LOOCV)</li>
        <li>Holdout Validation</li>
      </ul>
    <li>Model Performance Beyond Metrics</li>
      <ul>
        <li>Inference latency and throughput</li>
        <li>Model size and memory requirements</li>
        <li>Training time and computational cost</li>
        <li>Robustness to adversarial examples</li>
        <li>Fairness and bias metrics</li>
        <li>Explainability and interpretability scores</li>
      </ul>
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
        <li><a href="https://github.com/GokuMohandas/Made-With-ML">(Class) Made With ML</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>ML Pipelines</b></summary>
  <ul>
    <li>Pipeline Phases</li>
      <ul>
        <li>Data Preparation</li>
          <ul>
            <li>Data Extraction</li>
            <li>Data Analysis</li>
            <li>Data Preparation</li>
          </ul>
        <li>Model Development</li>
          <ul>
            <li>Model Training</li>
            <li>Model Evaluation</li>
            <li>Model Validation</li>
          </ul>
        <li>Model Serving</li>
          <ul>
            <li>Model Registry</li>
            <li>Model Prediction</li>
            <li>Model Performance Monitoring</li>
          </ul>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Python</b></summary>
  <ul>
    <li>Learn object-oriented programming (OOP) principles.</li>
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
  <summary><b>Distributed Computing & Big Data Processing</b></summary>
  <ul>
    <li>Leverage distributed computing for large-scale data processing.</li>
    <li>Use PySpark for writing scalable, Python-based ETL and analysis pipelines.</li>
  </ul>
</details>

<details>
  <summary><b>Data Warehouses & Lakehouses</b></summary>
  <ul>
    <li><a href="https://www.databricks.com/" target="_blank" rel="noopener noreferrer">Databricks</a></li>
    <li><a href="https://www.snowflake.com/" target="_blank" rel="noopener noreferrer">Snowflake</a></li>
    <li><a href="https://cloud.google.com/bigquery" target="_blank" rel="noopener noreferrer">GCP BigQuery</a></li>
    <li><a href="https://aws.amazon.com/redshift/" target="_blank" rel="noopener noreferrer">Amazon Redshift</a></li>
    <li><a href="https://azure.microsoft.com/en-us/products/synapse-analytics" target="_blank" rel="noopener noreferrer">Azure Synapse Analytics</a></li>
  </ul>
</details>


<!------------------ Section --------------------->

<br>

### Gen AI & Foundation Models
---
As generative AI and foundation models become central to modern AI products, PMs must understand their capabilities, limitations, costs, and operational requirements. Knowledge of transformer architectures, scaling laws, and fine-tuning techniques allows PMs to make strategic decisions about when to use pre-trained models versus building custom solutions, estimate compute costs and infrastructure needs, and set appropriate expectations for solution performance with executives. This expertise is crucial for navigating the rapidly evolving landscape of foundation models and ensuring products leverage the latest advances while maintaining cost efficiency and reliability.

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
    <li>Architecture & modeling</li>
      <ul>
        <li>Attention mechanism</li>
        <li>Positional encoding</li>
        <li>Tokenization & vector embeddings</li>
        <li>Decoder-only, encoder-only, and encoder-decoder architectures</li>
        <li>Key-Value (KV) Cache</li>
        <li>Hyperparameters</li>
          <ul>
            <li>Temperature</li>
            <li>Top-K</li>
            <li>Top-P</li>
          </ul>
        <li>Multi-head Latent Attention (MLA)</li>
        <li>Mixture-of-Experts (MoE)</li>
      </ul>
    <li>Resources:</li>
    <ul>
        <li><a href="https://www.youtube.com/watch?v=7xTGNNLPyMI&ab_channel=AndrejKarpathy">(Video) Deep Dive into LLMs like ChatGPT by Andrej Karpathy</a></li>
        <li><a href="https://arxiv.org/abs/1706.03762">(Paper) Attention is All You Need</a></li>
        <li><a href="https://www.youtube.com/watch?v=9vM4p9NN0Ts&ab_channel=StanfordOnline">(Class) Stanford CS229 - Machine Learning - Building Large Language Models (LLMs)</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Gen AI Scaling Laws</b></summary>
  <ul>
    <li>Pre-training Scaling (parameter, data, and training compute size)</li>
    <li>Post-training Scaling (RLHF)</li>
    <li>Test-time Scaling/Inference Scaling</li>
    <li>Multi-Agent Scaling</li>
  </ul>
</details>

<details>
  <summary><b>Diffusion Models</b></summary>
  <ul>
    <li>Fundamentals</li>
      <ul>
        <li>Forward diffusion process (adding noise to data)</li>
        <li>Reverse diffusion process (denoising to generate samples)</li>
        <li>Noise schedule and timesteps</li>
        <li>Score-based generative models</li>
      </ul>
    <li>Architecture & Techniques</li>
      <ul>
        <li>Denoising Diffusion Probabilistic Models (DDPM)</li>
        <li>Denoising Diffusion Implicit Models (DDIM)</li>
        <li>Latent Diffusion Models (LDM)</li>
        <li>Stable Diffusion architecture</li>
        <li>Classifier-free guidance</li>
        <li>Conditional generation (text-to-image, image-to-image)</li>
      </ul>
    <li>Applications</li>
      <ul>
        <li>Image generation</li>
        <li>Text-to-image synthesis</li>
        <li>Image inpainting and editing</li>
        <li>Super-resolution</li>
        <li>Audio generation</li>
        <li>Video generation</li>
      </ul>
    <li>Training & Optimization</li>
      <ul>
        <li>Training objectives (variational lower bound, score matching)</li>
        <li>Sampling strategies (DDPM, DDIM, DPM-Solver)</li>
        <li>Inference speed optimization</li>
        <li>Distillation techniques</li>
      </ul>
    <li>Resources</li>
      <ul>
        <li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">(Blog) What are Diffusion Models? by Lilian Weng</a></li>
        <li><a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">(Video) How Diffusion Models Work by AI Jason</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Generative Adversarial Networks (GAN)</b></summary>
  <ul>
    <li>Fundamentals</li>
      <ul>
        <li>Generator and discriminator networks</li>
        <li>Adversarial training process</li>
        <li>Minimax game formulation</li>
        <li>Nash equilibrium in GAN training</li>
        <li>Loss functions (adversarial loss, feature matching)</li>
      </ul>
    <li>Architecture Variants</li>
      <ul>
        <li>Deep Convolutional GAN (DCGAN)</li>
        <li>Progressive GAN (ProGAN)</li>
        <li>StyleGAN and StyleGAN2</li>
        <li>Conditional GAN (cGAN)</li>
        <li>Wasserstein GAN (WGAN)</li>
        <li>Least Squares GAN (LSGAN)</li>
        <li>CycleGAN (for image-to-image translation)</li>
        <li>Pix2Pix</li>
      </ul>
    <li>Training Challenges & Solutions</li>
      <ul>
        <li>Mode collapse</li>
        <li>Training instability</li>
        <li>Vanishing gradients</li>
        <li>Non-convergence issues</li>
        <li>Techniques for stable training (spectral normalization, gradient penalty)</li>
      </ul>
    <li>Applications</li>
      <ul>
        <li>Image generation</li>
        <li>Image-to-image translation</li>
        <li>Data augmentation</li>
        <li>Super-resolution</li>
      </ul>
    <li>Evaluation Metrics</li>
      <ul>
        <li>Inception Score (IS)</li>
        <li>Fréchet Inception Distance (FID)</li>
        <li>Kernel Inception Distance (KID)</li>
        <li>Precision and Recall for distributions</li>
      </ul>
    <li>Resources</li>
      <ul>
        <li><a href="https://www.deeplearning.ai/courses/generative-adversarial-networks-gans-specialization/" target="_blank" rel="noopener noreferrer">(Course) DeepLearning.AI - GAN Specialization</a></li>
        <li><a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noopener noreferrer">(Paper) Original GAN Paper (Goodfellow et al.)</a></li>
        <li><a href="https://arxiv.org/abs/1701.07875">(Paper) Wasserstein GAN by Arjovsky et al.</a></li>
        <li><a href="https://www.tensorflow.org/tutorials/generative/dcgan" target="_blank" rel="noopener noreferrer">(Tutorial) TensorFlow Deep Convolutional GAN Tutorial</a></li>
        <li><a href="https://www.youtube.com/watch?v=8L11aMN5KY8" target="_blank" rel="noopener noreferrer">(Video) A Friendly Introduction to GANs by Serrano Academy</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Fine-Tuning</b></summary>
  <ul>
    <li>Parameter-Efficient Fine-Tuning (PEFT)</li>
      <ul>
        <li>Low-Rank Adaptation (LoRA)</li>
        <li>QLoRA (Quantized LoRA)</li>
        <li>Adapter modules (Houlsby, Pfeiffer)</li>
        <li>Prefix tuning</li>
        <li>Prompt tuning</li>
        <li>P-tuning / P-tuning v2</li>
        <li>IA³ (Infused Adapter by Inhibiting and Amplifying)</li>
      </ul>
    <li>Quantization</li>
      <ul>
        <li>Quantized Low-Rank Adaptation (QLoRA)</li>
        <li>Post-training quantization (PTQ)</li>
        <li>Quantization-aware training (QAT)</li>
        <li>INT8 / INT4 / binary quantization</li>
        <li>GPTQ (GPT Quantization)</li>
        <li>AWQ (Activation-aware Weight Quantization)</li>
        <li>SmoothQuant</li>
        <li>BitsAndBytes</li>
      </ul>
    <li>Pruning</li>
      <ul>
        <li>Structured vs. unstructured pruning</li>
        <li>Magnitude-based pruning</li>
        <li>Movement pruning</li>
        <li>Wanda (Weight and Activation pruning)</li>
        <li>LLM-Pruner and similar methods</li>
      </ul>
    <li>Knowledge Distillation</li>
      <ul>
        <li>Teacher–student distillation</li>
        <li>Response distillation</li>
        <li>Feature / representation distillation</li>
      </ul>
    <li>Resources</li>
      <ul>
        <li><a href="https://arxiv.org/abs/2106.09685">(Paper) LoRA: Low-Rank Adaptation of Large Language Models by Hu et al.</a></li>
        <li><a href="https://arxiv.org/abs/2305.14314">(Paper) QLoRA: Efficient Finetuning of Quantized LLMs by Dettmers et al.</a></li>
        <li><a href="https://github.com/huggingface/peft">(Library) Hugging Face PEFT</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Benchmarks</b></summary>
  <ul>
    <li>Fundamentals</li>
      <ul>
        <li>What benchmarks measure (knowledge, reasoning, safety, coding, etc.)</li>
        <li>Leaderboards vs. arenas vs. LLM-as-judge</li>
        <li>Limitations of benchmarks (data contamination, narrow tasks, overfitting)</li>
        <li>When to trust benchmarks vs. real-world evaluation</li>
      </ul>
    <li>General Model Knowledge & Capability</li>
      <ul>
        <li>MMLU (Massive Multitask Language Understanding)</li>
        <li>HellaSwag</li>
        <li>TruthfulQA</li>
        <li>ARC (AI2 Reasoning Challenge)</li>
        <li>OpenBookQA</li>
        <li>Winogrande</li>
        <li>PIQA (Physical IQ)</li>
      </ul>
    <li>Reasoning Benchmarks</li>
      <ul>
        <li>GSM8K (math word problems)</li>
        <li>MATH</li>
        <li>HumanEval (code)</li>
        <li>Big-Bench (BIG-Bench Hard)</li>
        <li>DROP (discrete reasoning)</li>
        <li>CommonsenseQA</li>
        <li>StrategyQA</li>
      </ul>
    <li>Coding Benchmarks</li>
      <ul>
        <li>HumanEval</li>
        <li>MBPP (Mostly Basic Python Programming)</li>
        <li>DS-1000</li>
        <li>SWE-bench</li>
        <li>CodeContests</li>
        <li>MultiPL-E</li>
      </ul>
    <li>Safety & Alignment</li>
      <ul>
        <li>TruthfulQA (truthfulness)</li>
        <li>RealToxicityPrompts</li>
        <li>BBH (BIG-Bench Hard) safety subsets</li>
        <li>Red-teaming and adversarial benchmarks</li>
      </ul>
    <li>Multimodal & Long-Context</li>
      <ul>
        <li>MMMU, MMMU-Pro</li>
        <li>ChartQA, DocVQA</li>
        <li>Long-context benchmarks (Needle in a Haystack, etc.)</li>
      </ul>
    <li>Aggregate Suites & Leaderboards</li>
      <ul>
        <li>Open LLM Leaderboard (Hugging Face)</li>
        <li>LMSys Chatbot Arena</li>
        <li>MT-Bench</li>
        <li>AlpacaEval</li>
        <li>HELM (Holistic Evaluation of Language Models)</li>
      </ul>
    <li>Resources</li>
      <ul>
        <li><a href="https://www.latent.space/p/benchmarks-101">(Podcast) AI Fundamentals: Benchmarks 101</a></li>
        <li><a href="https://www.latent.space/p/benchmarks-201">(Podcast) Benchmarks 201: Why Leaderboards > Arenas >> LLM-as-Judge</a></li>
        <li><a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard">(Leaderboard) Open LLM Leaderboard</a></li>
        <li><a href="https://chat.lmsys.org/">(Arena) LMSys Chatbot Arena</a></li>
        <li><a href="https://arxiv.org/abs/2211.09110">(Paper) Holistic Evaluation of Language Models (HELM)</a></li>
        <li><a href="https://github.com/open-compass/opencompass">(Suite) OpenCompass Evaluation</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Evaluation</b></summary>
  <ul>
    <li>LLM evaluation metrics</li>
      <ul>
        <li>Statistical metrics</li>
          <ul>
            <li><a href="https://en.wikipedia.org/wiki/BLEU">BLEU</a></li>
            <li><a href="https://en.wikipedia.org/wiki/ROUGE_(metric)">ROUGE (Recall-Oriented Understudy for Gisting Evaluation)</a></li>
            <li><a href="https://en.wikipedia.org/wiki/METEOR">METEOR (Metric for Evaluation of Translation with Explicit Ordering)</a></li>
            <li><a href="https://en.wikipedia.org/wiki/Levenshtein_distance">Levenshtein Distance</a></li>
          </ul>
        <li>LLM-as-judge metrics</li>
      </ul>
    <li>Evaluation datasets</li>
      <ul>
        <li>Golden dataset</li>
        <li>Adversarial dataset</li>
        <li>Regression dataset</li>
      </ul>
    <li>LLM evaluation tools</li>
      <ul>
        <li><a href="https://docs.ragas.io">Ragas</a></li>
        <li>OpenAI Evals</li>
      </ul>
    <li>LLM-as-a-judge techniques</li>
      <ul>
        <li>Pairwise comparison</li>
        <li>Evaluation by criteria (reference free)</li>
        <li>Evaluation by criteria (reference-based)</li>
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
  <summary><b>AI Interpretability</b></summary>
  <ul>
    <li>Interpretability Methods</li>
      <ul>
        <li>Post-hoc Explainability</li>
        <li>Intrinsic Interpretability</li>
        <li>Mechanistic Interpretability</li>
      </ul>
    <li>Anthropic's Interpretability Team Publications</li>
      <ul>
        <li><a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html">Dictionary Learning</a></li>
        <li><a href="https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning">Monosemanticity</a></li>
        <li><a href="https://transformer-circuits.pub/2025/attribution-graphs/biology.html">Attributional Graphs</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>LLM Safety, Security, & Guardrails</b></summary>
  <ul>
    <li>Content filtering</li>
      <ul>
        <li><a href="https://cloud.google.com/security-command-center/docs/model-armor-overview">GCP Model Armor</a>
        <li>Prompt injection</li>
        </li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Prompt/Context Engineering</b></summary>
  <ul>
    <li>Chain-of-Thought (CoT) Prompting</li>
    <li>ReAct</li>
    <li>Tree-of-Thoughts (ToT) Prompting</li>
    <li><a href="https://platform.openai.com/docs/guides/text?api-mode=responses">(Article) OpenAI Prompting Guide</a></li>
    <li><a href="https://www.promptingguide.ai/">(Website) Prompt Engineering Guide by DAIR.AI</a></li>
  </ul>
</details>


<!------------------ Section --------------------->

<br>

### Retrieval Augmented Generation (RAG)
---
RAG has become the dominant pattern for building production LLM applications that need access to private or up-to-date information. PMs with RAG expertise can design systems that effectively combine retrieval and generation, choose appropriate vector databases and embedding strategies, and establish evaluation frameworks that measure both retrieval quality and response relevance. Understanding RAG is essential for building or evaluating AI products that go beyond simple chatbot interfaces to create intelligent systems that can reason over large knowledge bases and provide accurate, contextualized responses.

<details>
  <summary><b>RAG Fundamentals</b></summary>
  <ul>
    <li>Vector embeddings</li>
    <li>Chunking</li>
    <li>Hybrid retrieval</li>
    <li>General resources</li>
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
    <li>RAG evaluation metrics</li>
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
      <li><a href="https://docs.ragas.io/en/stable/">Ragas</a></li>
    </ul>
  </ul>
</details>

<details>
  <summary><b>Vector Databases</b></summary>
  <ul>
    <li>Vector search prototyping libraries</li>
      <ul>
        <li><a href="https://faiss.ai/">FAISS</a></li>
        <li><a href="https://github.com/nmslib/hnswlib">HNSWlib</a></li>
      </ul>
    <li>Production databases</li>
      <ul>
        <li><a href="https://www.pinecone.io/">Pinecone</a></li>
        <li><a href="https://weaviate.io/">Weaviate</a></li>
        <li><a href="https://www.trychroma.com/">Chroma</a></li>
        <li><a href="https://www.elastic.co/elasticsearch">Elasticsearch</a></li>
        <li><a href="https://milvus.io/">Milvus</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Agentic RAG</b></summary>
</details>



<!------------------ Section --------------------->

<br>

### Agentic AI
---
Agentic AI represents the next evolution of AI systems, moving from single-turn interactions to autonomous agents that can plan, reason, and take actions across multiple steps. PMs need to understand agent frameworks, design patterns, and evaluation methods to build products that can handle complex, multi-step workflows. This knowledge is critical for designing agent architectures that balance autonomy with control, ensuring agents can operate safely and effectively in production environments while delivering on ambitious product visions that require sophisticated reasoning and tool use.

<details>
  <summary><b>Agent Fundamentals</b></summary>
  <ul>
    <li>Resources</li>
      <ul>
        <li><a href="https://www.kaggle.com/whitepaper-agents">(White Paper) Google Agents White Paper by Julia Wiesinger et al.</a></li>
        <li><a href="https://www.kaggle.com/whitepaper-agent-companion">(White Paper) Google Agents Companion by Antonio Gulli et al.</a></li>
        <li><a href="https://arxiv.org/abs/2505.10468">(Paper) AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges</a></li>
        <li><a href="https://arxiv.org/abs/2210.03629">(Paper) ReAct: Synergizing Reasoning and Acting in Language Models by Shunyu Yao et al.</a></li>
        <li><a href="https://huggingface.co/learn/agents-course/en/unit0/introduction">(Course) HuggingFace AI Agents Course</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Agent Evaluation</b></summary>
  <ul>
    <li>Resources</li>
      <ul>
        <li><a href="https://arxiv.org/abs/2410.10934">(Paper) Agent-as-a-Judge: Evaluate Agents with Agents</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Agent Frameworks</b></summary>
  <ul>
    <li><a href="https://www.langchain.com/">LangChain</a>: open-source framework for building agents</li>
    <li><a href="https://www.langchain.com/">LangGraph</a>: a newer-graph-based library within the LangChain ecosystem designed for creating complex, stateful, and multi-agent workflows with explicit state management and the ability to handle loops and cycles.</li>
      <ul>
        <li><a href="https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/">(Class) AI Agents in LangGraph by DeepLearning.AI</a></li>
      </ul>
    <li><a href="https://www.llamaindex.ai/">LlamaIndex</a>: specializes in efficiently indexing and querying large datasets for Retrieval Augmented Generation (RAG) applications</li>
      <ul>
        <li><a href="https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/">(Class) Building Agentic RAG with LlamaIndex</a></li>
      </ul>
    <li><a href="https://openai.github.io/openai-agents-python/">OpenAI Agent SDK</a>: OpenAI's framework for building agentic AI apps in a lightweight, easy-to-use package with few abstractions. It's a production-ready upgrade of their previous experimentatal framework (Swarm).</li>
    <li><a href="https://google.github.io/adk-docs/">Google Agent Development Kit (ADK)</a></li>
      <ul>
        <li><a href="https://codelabs.developers.google.com/onramp/instructions#0">(Code Lab) ADK Crash Course - From Beginner To Expert</a></li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>Agent Protocols</b></summary>
  <ul>
    <li><a href="https://modelcontextprotocol.io/docs/getting-started/intro">Anthropic's Model Context Protocol (MCP)</a></li>
    <li><a href="https://a2a-protocol.org/latest/">Google's Agent-2-Agent (A2A)</a></li>
    <li><a href="https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol">Google's Agent Payments Protocol (AP2)</a></li> 
  </ul>
</details>

<details>
  <summary><b>AI Integrated Development Environments (IDEs)</b></summary>
  <ul>
    <li><a href="https://cursor.com/agents">Cursor</a></li>
    <li><a href="https://windsurf.com/">Windsurf</a></li>
    <li><a href="https://replit.com/">Replit</a></li>
    <li><a href="https://openai.com/codex/">OpenAI Codex</a></li>
    <li><a href="https://claude.com/product/claude-code">Claude Code</a></li>
    <li><a href="https://codeassist.google/">Gemini Code Assist</a></li>
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


<!------------------ Section --------------------->

<br>

### General Technical Skills
---
While AI PMs don't need to be expert engineers, technical literacy across cloud infrastructure, data pipelines, APIs, and system architecture is essential for making informed product decisions and effectively collaborating with engineering teams. Understanding these technical domains enables PMs to estimate development effort accurately, identify technical risks early, design scalable product architectures, and make trade-off decisions between different technical approaches. This foundation ensures PMs can translate between business requirements and technical implementation, bridging the gap between stakeholders and engineering teams.

<details>
  <summary><b>Public Cloud Infrastructure</b></summary>
  <ul>
    <li><a href="https://cloud.google.com/">Google Cloud Platform (GCP)</a></li>
    <li><a href="https://azure.microsoft.com/en-us/">Microsoft Azure</a></li>
    <li><a href="https://aws.amazon.com/">Amazon Web Services (AWS)</a></li>
  </ul>
</details>

<details>
  <summary><b>Cloud FinOps</b></summary>
  <ul>
    <li>Manage cloud financial operations to maximize efficiency and optimize cost.</li>
  </ul>
</details>

<details>
  <summary><b>Data Pipelines</b></summary>
  <ul>
    <li>Pipeline Technologies</li>
      <ul>
        <li>Apache Airflow</li>
          <ul>
            <li>GCP Composer</li>
            <li>Amazon Managed Workflows for Apache Airflow (MWAA)</li>
            <li>Azure Workflow Orchestration Manager</li>
          </ul>
        <li>Apache Beam</li>
          <ul>
            <li>GCP Dataflow</li>
          </ul>
        <li>AWS Glue</li>
        <li>Apache Kafka</li>
        <li>Kubeflow Pipelines (KFP)</li>
        <li>TensorFlow Extended (TFX)</li>
      </ul>
  </ul>
</details>

<details>
  <summary><b>API and Backend Skills</b></summary>
  <ul>
    <li>Develop backends with FastAPI or Flask</li>
    <li>Implement REST or GraphQL streaming endpoints for AI services</li>
    <li>Design authentication and rate-limiting systems</li>
    <li>Build WebSocket implementations for real-time AI interactions</li>
  </ul>
</details>

<details>
  <summary><b>Data Validation</b></summary>
  <ul>
    <li><a href="https://docs.pydantic.dev/latest/">Pydantic</a></li>
  </ul>
</details>

<details>
  <summary><b>Architecture Concepts</b></summary>
  <ul>
    <li><ins>Load Balancing</ins>: distributes incoming traffic across multiple servers to improve reliability and performance.</li>
    <li><ins>Caching</ins>: stores frequently accessed data in memory to reduce latency and database load.</li>
    <li><ins>Content Delivery Networks</ins>: store static assets across edge servers to reduce latency for users.</li>
    <li><ins>Message Queue</ins>: decouples components by allowing producers to enqueue messages that consumers process asynchronously.</li>
    <li><ins>Publish-Subscribe</ins>: enables multiple consumers to receive messages from a shared topic.</li>
    <li><ins>API Gateway</ins>: acts as a single entry point for client requests and handles routing to backend services.</li>
    <li><ins>Circuit Breaker</ins>: monitors downstream service calls and stops attempts when failures exceed a defined threshold.</li>
    <li><ins>Service Discovery</ins>: automatically tracks available service instances so consumers can locate providers.</li>
    <li><ins>Sharding</ins>: splits large datasets across multiple nodes based on a shard key.</li>
    <li><ins>Rate Limiting</ins>: controls the number of requests a client can make within a given time window.</li>
    <li><ins>Consistent Hashing</ins>: distributes data across nodes while minimizing reorganization when nodes are added or removed.</li>
    <li><ins>Auto Scaling</ins>: automatically adds or removes compute resources based on system metrics.</li>
  </ul>
</details>


<!------------------ Section --------------------->

<br>

### General Product Management Skills
---
Core product management skills remain fundamental for AI PMs, but they take on new dimensions when applied to AI products. Project management frameworks help structure the iterative, experimental nature of AI development. Metrics and analytics become more complex when dealing with probabilistic systems where success isn't binary. Stakeholder management requires explaining uncertain outcomes and managing expectations around model performance. These foundational PM skills are essential for delivering AI products that not only work technically but also create genuine user value and business impact.

<details>
  <summary><b>Project Management Frameworks</b></summary>
  <ul>
    <li><ins>Waterfall</ins>: A traditional, sequential approach where each project phase is completed before the next begins. Each phase has specific deliverables and a review process, making it suitable for projects with clearly defined requirements and predictable outcomes. However, it offers limited flexibility for changes once a phase is complete.</li>
    <li><ins>Agile</ins>: An iterative and incremental approach, suitable for projects with evolving requirements</li>
      <ul>
        <li><ins>Scrum</ins>: structured roles, sprints, and ceremonies.</li>
        <li><ins>Kanban</ins>: visual flow-based system emphasizing WIP limits and continuous delivery.</li>
      </ul>
    <li><ins>Lean</ins>: </li>
  </ul>
</details>

<details>
  <summary><b>Continuous Integration, Testing, and Delivery (CI/CT/CD)</b></summary>
  <ul>
    <li>Concepts</li>
      <ul>
        <li><ins>Continuous Integration (CI)</ins>: Devs frequently merge code changes into a central repo, triggering automated builds to find integration issues quickly.</li>
        <li><ins>Continuous Testing (CT)</ins>: Automated tests (unit, integration, etc.) run continuously throughout the pipeline, verifying code quality and preventing bugs from reaching production.</li>
        <li><ins>Continuous Delivery (CD)</ins>: automatically prepares code for release, ensuring it's always in a deployable state, ready to be pushed to production with a single click.</li>
      </ul>
    <li>Tools</li>
      <ul>
        <li>Jenkins</li>
        <li>GitHub Actions</li>
        <li>AWS Code Pipeline</li>
        <li>GCP Cloud Build/Cloud Deploy</li>
        <li>Azure Pipelines</li>
      </ul>
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
  <summary><b>Product Metrics</b></summary>
  <ul>
    <li>User Adoption & Engagement</li>
      <ul>
        <li><ins>Daily/Monthly Active Users (DAU/MAU)</ins>: the number of unique users who interact with the system daily/monthly</li>
        <li><ins>DAU/MAU Stickiness</ins>: measures how frequently users return to the product (higher ratios = better)</li>
        <li><ins>Feature Adoption Rate</ins>: percentage of users who use a specific feature within a given time period</li>
      </ul>
    <li>Customer Satisfaction</li>
      <ul>
        <li><ins>Net Prompter Score (NPS)</ins>: measures customer loyalty and satisfaction by asking users the likelihood of product recommendations</li>
        <li><ins>Customer Satisfaction Score (CSAT)</ins>: measures customer happiness with a specific feature or interaction</li>
        <li><ins>System Usability Scale (SUS)</ins>: measures the perceived usability of a product or system (range: 0-100 with higher scores indicating better usability). Importanlty, SUS doesn't measure the satisfaction with the product, so SUS + NPS is a good strategy.</li>
        <li><ins>Support Ticket Ratio</ins>: number of support tickets per user. A high number can signal product friction, bugs, or adoption issues.</li>
      </ul>
    <li>Retention & Churn</li>
      <ul>
        <li><ins>Churn Rate</ins>: percentage of customers who stop using the product within a given timeframe</li>
        <li><ins>Retention Rate</ins>: percentage of customers who continue to use the product over a specific period. Retention is important as it's generally more cost efficient than new customer acquisitions</li>
      </ul>
    <li>Financial Performance</li>
      <ul>
        <li><ins>Customer Acquisition Cost (CAC)</ins>: </li>
        <li><ins>Customer Lifetime Value (CLTV)</ins>: </li>
        <li><ins>Monthly Recurring Revenue (MRR)</ins>: </li>
        <li><ins>Annual Recurring Revenue (ARR)</ins>: </li>
        <li><ins>Average Revenue Per User (ARPU)</ins>: </li>
      </ul>
    <li>Product Performance & Delivery</li>
  </ul>
</details>

<details>
  <summary><b>Robust Documentation</b></summary>
  <ul>
    <li>Ensure product documentation is clear, current, and accessible to cross-functional teams.</li>
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

<details>
  <summary><b>User Journeys</b></summary>
  <ul>
    <li>Define clear user journeys aligned with a strategic AI product philosophy and a north star metric.</li>
  </ul>
</details>

<details>
  <summary><b>Time Management and Productivity</b></summary>
  <ul>
    <li><ins>Eisenhower Matrix</ins>: organizing tasks into four quadrants, divided up by “urgency” on one axis and “importance” on the other</li>
    <li><ins>Covey Matrix</ins>: uses the same grid as the Eisenhower Matrix, but focuses on long-term planning instead of on daily tasks.</li>
    <li><ins>Pomodoro Technique</ins>: Pomodoro means “tomato” in Italian, and this method is named for the tomato-shaped timers originally used for it. The technique is simple to follow, set a timer for 25 minutes, work, and then set it for a five-minute break. Repeat the process four times.</li>
    <li><ins>Flowtime Technique</ins>: Sometimes called the “Flomodoro,” this method tries to help workers break through to a flow state. Users set a plan for the day and divide up their goals into small, manageable tasks. From there, they work without timers, taking breaks as needed. It’s essentially a deconstructed version of the Pomodoro Technique</li>
    <li><ins>Eat the Frog</ins>: Mark Twain said if you have to eat a frog, do it first thing in the morning. This approach to work is straightforward: Always do the most difficult and dreaded task first. Avoid multitasking and home in on the single task early. Then, the rest of the day will seem breezy by comparison.</li>
  </ul>
</details>

<details>
  <summary><b>Laws & Axioms</b></summary>
  <ul>
    <li><ins>Goodhart's Law</ins>: when a measure becomes a target, it ceases to be a good measure</li>
    <li>The Mythical Man Month</li>
      <ul>
        <li><ins>Brooks's Law</ins>: The central thesis that adding more people to a late project increases the delay due to the exponential growth in communication paths and ramp-up time</li>
        <li><ins>Man-Month Myth</ins>: The idea that a "man-month" is a standard unit of work is deceptive; men and months are not interchangeable because tasks aren't always perfectly divisible</li>
      </ul>
    <li><ins>The Lindy Effect</ins>: a technology principle that suggests that the future life expectancy of a non-perishable technology, system, or idea is proportional to its current age. Coined by Nassim Taleb, this principle argues that the longer a technology has already survived, the higher the probability it will continue to endure, as it has proven its resilience and robustness over time</li>
    <li><ins>Hersey-Blanchard Model</ins>:</li> 
  </ul>
</details>


<!------------------ Section --------------------->

<br>

### AI Specific Product Skills
---
AI-specific product skills distinguish exceptional AI PMs from traditional PMs. AI product sense involves understanding when AI is the right solution versus when simpler approaches would be more effective. Experiment design for AI products requires knowledge of A/B testing with probabilistic systems, evaluation metrics that capture model quality, and iterative development cycles that account for model training and refinement. Market insight in the AI space is crucial given the rapid pace of innovation, where new models and capabilities emerge weekly and can fundamentally change what's possible. These skills enable PMs to navigate the unique challenges of building and launching AI products successfully.

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
  <summary><b>AI Market Insight</b></summary>
  <ul>
    <li>Build a deep understanding of the AI market, its competitive landscape, and emerging trends</li>
  </ul>
</details>


<br>

## Newsletters

---

<br>

### Tech News
---
- [Data Points](https://www.deeplearning.ai/the-batch/tag/data-points/): a twice‑weekly series from DeepLearning.AI of the most important AI tools, model releases, research findings, and industry developments
- [Daily Zaps](https://www.dailyzaps.com/): a daily high-level tech news roundup that trends more business than technical
- [The Download from MIT Technology Review](https://www.technologyreview.com/topic/download-newsletter/): a daily high-level tech news roundup from MIT Tech Review
- [Tech Brew](https://www.emergingtechbrew.com/): a punchy, daily roundup of general technology news from the editors of the popular Morning Brew newsletter.

<br>

### Cloud Developer Programs
---
- [Google Developer Program](https://developers.google.com/newsletter): stay up to date with the latest GCP releases and features
- [Microsoft.Source newsletter](https://info.microsoft.com/ww-landing-sign-up-for-the-microsoft-source-newsletter.html): the curated monthly developer community newsletter provides the latest articles, documentation, and events.
- [AWS Builder Center](https://builder.aws.com/): Connect with other builders, share solutions, influence AWS product development, and access useful content.

<br>

### Engineering Deep Dives
---
- [TheSequence](https://thesequence.substack.com/): A weekly series that does technical deep dives on the latest AI/ML techniques
- [The Batch @ DeepLearning.AI](https://www.deeplearning.ai/the-batch/): a weekly deep dive from Stanford Professor Andrew Ng
- [The MLOps Newsletter](https://mlops.substack.com/): technical with a specific focus on MLOps
- [The Variable](https://towardsdatascience.com/category/the-variable/): a curated list of articles/tutorials from Towards Data Science
- [Turing Post](https://www.turingpost.com/subscribe?ref=WAGU23hEVa): a weekly newsletter by Ksenia Se that covers AI and ML curated summaries of hundreds of industry developments, research insights, and historical context
- [SwirlAI](https://www.newsletter.swirlai.com/): MLOps and data engineering focused newsletter with great visualizations
- [ByteByteGo](): a weekly newsletter by Alex Xu that delivers concise system‑design deep dives and foundational tech explainers on complex distributed systems topics like Kubernetes, databases, CI/CD, and API design


<br>


## Podcasts
---
- [Practical AI by Changelog](https://podcasts.apple.com/us/podcast/practical-ai/id1406537385)
- [Inference by Turing Post](https://www.youtube.com/playlist?list=PLRRoCwK1ZTNCAZXXOswpIYQqzMgT4swsI)
- [Latent Space: The AI Engineer Podcast](https://www.latent.space/podcast)
- [NVIDIA AI Podcast](https://ai-podcast.nvidia.com/)
- [Hard Fork by the NYT](https://www.nytimes.com/column/hard-fork)
- [The a16z Podcast by Andreessen Horowitz](https://a16z.com/podcasts/a16z-podcast/)


<br>


## People
---
- The "Godfathers of AI"
  - [Yann LeCun](http://yann.lecun.com/): Chief AI Scientist at Meta and a pioneer in optical character recognition (OCR) and convolutional neural networks (CNN). A Turing Award winner and one of the three "Godfathers of AI".
  - [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton): University Professor Emeritus at the University of Toronto and former Google Brain lead. Popularizer of backpropagation, AlexNet, and deep learning. 2024 Nobel Prize winner in Physics for machine learning with artificial neural networks. Also well known as a mentor with his former graduate students being Alex Krizhevsky, Ilya Sutskever, Yann LeCun, and many other luminaries. In May 2023, Hinton resigned from Google and started speaking out against the dangers of AI.
  - [Yoshua Bengio](https://yoshuabengio.org/): co-Turing Award winne in 2018 with Yann LeCun and Geoffrey Hinton for his work on deep learning. Bengio is the most-cited computer scientist globally (by both total citations and by h-index), and the most-cited living scientist across all fields (by total citations).
- [Demis Hassabis](https://en.wikipedia.org/wiki/Demis_Hassabis): CEO and co-founder of Google DeepMind. He was jointly awarded the Nobel Prize in Chemistry in 2024 for his work on AlphaFold and protein structure prediction.
- [Mustafa Suleyman](https://en.wikipedia.org/wiki/Mustafa_Suleyman): CEO of Microsoft AI and former head of applied AI at Google DeepMind
- [Andrej Karpathy](https://karpathy.ai/): Former director of Autopilot at Tesla, co-founder of OpenAI, and prolific AI educator.
- [Fei-Fei Li](https://en.wikipedia.org/wiki/Fei-Fei_Li): Stanford CS professor, co-director of the Stanford Institute for Human-Centered AI, inventor of ImageNet, and former Chief Scientist of AI/ML at GCP.
- [Eugene Yan](https://eugeneyan.com/subscribe): ML, RecSys, LLMs, and engineering
- [Andrew Ng](https://www.andrewng.org/): founder of Coursera, DeepLearning.AI, Stanford AI computer science professor, and neural network pioneer


<br>

## Books
---

<br>

### Pop Tech
---
Narrative and investigative books about major tech companies, case studies, and industry shifts. These works build intuition about how technology intersects with power, markets, and society; helping PMs think strategically beyond the model and anticipate real-world consequences.


#### Geopolitics
* [Apple in China: The Capture of the World's Greatest Company](https://www.simonandschuster.com/books/Apple-in-China/Patrick-McGee/9781668053379) by Patrick McGee
* [Chip War: The Fight for the World's Most Critical Technology](https://www.simonandschuster.com/books/Chip-War/Chris-Miller/9781982172015) by Chris Miller
* [AI Valley: Microsoft, Google, and the Trillion-Dollar Race to Cash In on Artificial Intelligence – A Definitive Insider Chronicle of the Breakthroughs Redefining Our World](https://bookriot.com/books/ai-valley-microsoft-google-and-the-trillion-dollar-race-to-cash-in-on-artificial-intelligence/) by Gary Rivlin


#### Artificial General Intelligence (AGI)
* [The Path to AGI: Artificial General Intelligence](https://technicspub.com/path-to-agi/) by John K. Thompson
* [The Intelligence Explosion: When AI Beats Humans At Everything](https://us.macmillan.com/books/9781250355027/theintelligenceexplosion/) by James Barrat
* [Superagency: What Could Possibly Go Right with Our AI Future](https://www.superagency.ai/) by Reid Hoffman and Greg Beato


#### Biographies & Memoirs
* [The Nvidia Way: Jensen Huang and the Making of a Tech Giant](https://wwnorton.com/books/the-nvidia-way) by Tae Kim
* [Source Code: My Beginnings](https://en.wikipedia.org/wiki/Source_Code_(memoir)) by Bill Gates


<br>

### Product Management
---
Foundational texts on building and scaling software products. They focus on execution, tradeoffs, and organizational dynamics, grounding AI product work in timeless principles for managing complexity and uncertainty.

* [The Mythical Man-Month: Essays on Software Engineering](https://en.wikipedia.org/wiki/The_Mythical_Man-Month) by Frederick Brooks


<br>

### Textbooks & Technical Writing
---
Technical books covering statistics, data analysis, and machine learning fundamentals. They give AI Product Managers the literacy needed to collaborate effectively with technical teams and make informed decisions about model capabilities and limitations.

#### Probability & Statistics
* [Practical Statistics for Data Scientists](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) by Peter Bruce

#### Math
* [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf) by Deisenroth, Faisal, and Ong

#### Programming
* Python
  * [Python 3 Object Oriented Programming](https://a.co/d/fvAmkN1) by Steven F. Lott and Dusty Phillips
  * [Python for Data Analysis, Data Wrangling with pandas, NumPy, & Jupyter](https://wesmckinney.com/book/) by Wes McKinney
* Rust
  * [The Rust Programming Language](https://doc.rust-lang.org/book/#the-rust-programming-language) by Steve Klabnik, Carol Nichols, and Chris Krycho

#### Cloud Engineering
* Google Cloud Platform (GCP)
  * [Official Google Cloud Certified Associate Cloud Engineer Study Guide](https://www.google.com/books/edition/Official_Google_Cloud_Certified_Associat/eNuMDwAAQBAJ?hl=en&gbpv=0) by Dan Sullivan
  * [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.oreilly.com/library/view/official-google-cloud/9781119944461/) by Mona Mona, Pratap Ramamurthy
* Amazon Web Services (AWS)
* Microsoft Azure

#### ML & Deep Learning
* [Deep Learning](https://www.deeplearningbook.org) by Ian Goodfellow
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) by Aurélien Géron

#### Gen AI
* [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
* [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952) by Jay Alammar

#### Natural Language Processing
* [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789) by Lewis Tunstall

#### AI Engineering
* [AI Engineering: Building Applications with Foundation Models](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) by Chip Huyen
* [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen
# Hi, I'm Harshita Sayala! 

### About Me

A **detail-oriented Master's student** in **Data Analytics** at SJSU, I possess a **strong foundation** in **data warehousing, ETL workflows, machine learning, and data visualization**.

I'm deeply passionate about leveraging data to **extract meaningful insights** and **build intelligent solutions**, with interests spanning:
* **Data Science**
* **Machine Learning**
* **Deep Learning**
* **Generative AI**
* **Data Analytics**
* **Data Warehousing**

I thrive on tackling **complex data challenges** through **collaborative problem-solving** and **clear communication**, always eager to apply my **analytical and technical expertise** to drive **innovative and impactful business outcomes**.

---

### Technical Skills

Here's a breakdown of my technical toolkit:

* **Programming Languages:** Python, SQL, C, Java, HTML, CSS, JavaScript
* **Data Science & Machine Learning:**
    * **Libraries/Frameworks:** Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow/Keras
    * **Modeling/Concepts:** Regression Analysis, Hypothesis Testing, Predictive Modeling, Time-Series Forecasting (SARIMA, LSTM), Recommendation Systems (Collaborative Filtering, Content-Based Filtering)
* **Natural Language Processing (NLP) & Large Language Models (LLMs):**
    * **Concepts/Tools:** NLTK, BERT, TF-IDF
    * **LLM Specific:** LangChain, SentenceTransformers, OpenAI GPT-3.5, FAISS, RAG Pipeline
* **Data Engineering:** Apache Airflow, dbt (Data Build Tool), Apache Spark, Hadoop, Snowflake, MySQL, NoSQL, MongoDB, MongoDB Atlas
* **Data Analysis & Visualization:** Tableau, Power BI, Superset, Preset, Matplotlib, Seaborn, Streamlit
* **Cloud Platforms:** AWS (Amplify, EC2), Google Cloud Platform (GCP), Microsoft Azure
* **Containers & Deployment:** Docker, Kubernetes
* **Version Control & Development Tools:** Git, GitHub, Google Colab
* **Testing Tools & Technologies:** LoadRunner, Selenium, Jenkins, BDD Cucumber, JUnit, TestNG
* **SDLC & Project Management:** JIRA, Confluence, Trello, Lucidchart
* **Web Technologies:** Node.js, React
* **Productivity Tools:** Microsoft Office Suite (Word, Excel, PowerPoint, Outlook), Google Workspace (Docs, Sheets, Slides, Drive)

---
### Current Research
Developing LLM guardrails and hallucination prevention techniques using self-reflective, multi-agent frameworks for safe and reliable generative AI.

---

### My Featured Projects

Here are some of my key projects, demonstrating my expertise in data analytics, AI/Machine learning, datascience, software engineering and data engineering. 

####  [SpartanGuard - Enterprise LLM Guardrail System]
* **Description:** Built a modular five-phase safety framework that wraps any **enterprise LLM** in dual guardrail layers to catch "compliance hallucinations" - authoritative-sounding but false regulatory claims(fabricated standards, omitted exceptions, jurisdiction errors) in regulated domains like healthcare and finance. Combined an **input Gateway(PII, jailbreak, prompt-injection classifiers)** with an output **Multi-Agent Debate(MAD)** verifier and a self-improving **GRPO feedback** loop, all exposed through a React compliance dashboard.
* **Key Achievements:** Fine-tuned 5 models using **LoRA/QLoRA** on 235K+ training samples, achieving 99.23% **prompt-injection recall (0.77% false-negative rate, 0.9831 security score), 99.90% jailbreak-detection accuracy(F1 0.9990, ROC-AUC 1.0) and 0.967 micro-F1 PII NER** across 56 entity types with 0% malformed JSON.
Designed a **Multi-Agent Debate pipeline(Verifier vs. adversarial Skeptic + blind Judge)** grounded on a 4,664 chunk regulatory **RAG** corpus with a confidence-scoring engine, routing precision(judge-heavy config) and an unconditional HARD_BLOCK on fabricated material claims.
Implemented a **GRPO feedback loop** using a Brier proper-scoring reward that cut the Brier score by 98.3% (0.3397 → 0.0056) for confidence calibration and fine-tuned a domain embedding model to 94.9% Recall@1, beating off the shelf baselines.
* **Technologies:**  `PyTorch`, `HuggingFace Transformers`, `PEFT(LoRA/QLoRA)`, `TRL(GRPO)`, `Qwen2.5/Qwen3`, `Llama-3.1-8B`, `Qdrant`, `LangGraph/LangChain`, `vLLM`, `Ollama`, `FastAPI`, `React`, `DeepEval`, `Sentence-Transformers`, `Docker`

####  [Acoustic Species Identification with Hybrid Deep Learning]
* **Description:** Built a hybrid **deep learning pipeline** for the **BirdCLEF+ 2026 Kaggle competition** that identifies animal species from one-minute environmental soundscape recordings in the Brazilian Pantanal, framed as **multi-label classification** across 234 species. Combined pretrained bioacoustic embeddings, state space sequence modeling, and sound event detection into a CPU-only ensemble that adapts noisy real-world ecology audio to accurate species predictions.
* **Key Achievements:** Engineered a **multi-branch ensemble (Perch backbone + ProtoSSM + ResidualSSM + MLP probes + distilled SED)** over 234 species across birds, amphibians, insects, mammals, and reptiles, segmenting **60-second soundscapes** into **12 five-second windows** and achieving ~0.94 macro ROC-AUC on the public leaderboard. Leveraged the pretrained Perch model to extract 1536-dimensional embeddings, directly mapping 203 of 234 classes to its logits, while a bidirectional selective-SSM ProtoSSM (~723K parameters) modeled temporal dependencies and a **ResidualSSM** (~176K parameters, val MSE ≈ 0.024) applied second-pass error correction.
Improved performance from a 0.85 Perch-only baseline to 0.94 through incremental sequence modeling, residual correction, event detection, and a per-class 60/40 rank-blending ensemble, all running under the competition's CPU-only, no-internet, sub-90-minute constraints.
* **Technologies:**  `PyTorch`, `TensorFlow`, `Perch`, `State Space Models (Mamba-style SSM)`, `Sound Event Detection`, `Mel Spectrograms`, `Multi-Head Attention`, `Ensemble Learning`

####  [Multimodal Retrieval-Augmented Generation]
* **Description:** Built a **multimodal RAG system** that answers questions over a macroeconomics PDF by jointly retrieving text, figures, and tables, then generating grounded, **hallucination-free answers** tied to source evidence. Combined dense and sparse retrieval with cross-encoder reranking to maximize retrieval precision, earning a spot on the Kaggle class leaderboard.
* **Key Achievements:** Engineered a multimodal ingestion pipeline using **PyMuPDF** for text extraction, 180-word overlapping chunks (40-word overlap), and regex-based caption detection to extract figures/tables as separate retrievable objects.
Implemented hybrid retrieval combining dense embeddings (all-MiniLM-L6-v2) and BM25 sparse search (weighted 0.7 / 0.3), stored in a FAISS vector database with separate indexes for text, pages, and figures, plus cross-encoder reranking (ms-marco-MiniLM-L-6-v2).
Delivered fully grounded answers across all 15 evaluation queries in the required CSV schema with accurate figure/table references, achieving a leaderboard score of 0.43.
* **Technologies:**  `Python`, `RAG`, `FAISS`, `Sentence-Transformers`, `BM25`, `Cross-Encoder`, `PyMuPDF`, `Hugging Face`, `Multimodal Retrieval`

####  [Fine-Tuning Diffusion Models for Domain-Specific Image Generation]
* **Description:** Fine-tuned **Stable Diffusion 1.5** on a Pokémon image-caption dataset to compare **LoRA** against full **UNet** fine-tuning across image quality, domain adaptation, and compute cost. Evaluated all three models (base, LoRA, full) on identical prompts using both quantitative metrics and human ratings to identify the best efficiency-quality trade-off.
* **Key Achievements:** Fine-tuned **Stable Diffusion 1.5** on 833 image-caption pairs at 512×512 using two strategies - **LoRA (attention-layer adapters) and full UNet** fine-tuning — under matched prompts and resolution for fair comparison.
Cut peak GPU memory by ~50% with LoRA (9.79 GB vs 19.77 GB for full fine-tuning) while achieving the highest domain-authenticity score (5.0/5) in human evaluation.
Benchmarked all models with **Inception Score and CLIP similarity** (base IS 1.695, LoRA 1.663, full 1.610), showing full fine-tuning's lower diversity signaled overfitting and establishing LoRA as the optimal efficiency-quality balance.
* **Technologies:**  `Python`, `PyTorch`, `Stable Diffusion 1.5`, `LoRA, Hugging Face Diffusers`, `CLIP`, `Inception Score`, `Diffusion Models`

####  [Multimodal Multi-Agent System for Image Understanding & Generation](https://github.com/harshitasayala10/agentlens)
* **Description:** Designed a **four-agent pipeline** that takes an image plus a natural-language instruction, then captions, refines, generates, and self-critiques the output to decide whether it should be accepted or revised. Built around typed data contracts and per-step audit logs so each agent is modular, swappable, and fully debuggable.
* **Key Achievements:** Architected a **4-agent pipeline (Vision, Prompt, Generation, Critique)** using Claude Sonnet 4.5 for **vision/language and Stable Diffusion 1.5** (local diffusers) for generation, communicating via typed Python dataclasses with per-agent JSON audit logs.
Built a two-stage **self-critique agent** combining open_clip (ViT-B-32) dual similarity scores (image↔image and image↔text) with LLM-based scoring, plus a CLIP-only fallback so the system always returns a verdict.
Validated end-to-end across 3 use cases with a 5-criterion human-evaluation rubric, achieving 100% (5/5) human-agent agreement on accept/revise decisions, including correctly flagging a flawed watercolor output for revision.
* **Technologies:**  `Python`, `Claude Sonnet 4.5`, `Stable Diffusion 1.5`, `open_clip`, `Hugging Face Diffusers`, `Multi-Agent Systems`, `CLIP`

####  [GPT-Style LLM from Scratch - TinyStories]
* **Description:** Designed and trained a **GPT-style autoregressive language model** entirely from scratch on the TinyStories dataset to learn **character-level patterns** and generate short narrative text. Implemented the full **Transformer stack** - token/positional embeddings, masked multi-head self-attention, feed-forward blocks and a language-modeling head - and analyzed common sequence-generation failure modes.
* **Key Achievements:** Built a **character-level GPT from scratch** with 256-token context windows, training on 100K samples and validating on 10K samples using cross-entropy loss with **LR warm-up** over 15 epochs. Achieved stable convergence with training loss 0.77→0.73 and validation loss 0.74→0.71, indicating minimal overfitting and good generalization.
* Conducted structured failure analysis across 3 categories(broken grammar, abrupt cutoffs, coherence drift), tracing each to character-level next-token prediction and limited context window.
* **Technologies:**  `Python`, `PyTorch`, `Transformers`, `Self-Attention`, `Matplotlib`

####  [IMDB Sentiment Analysis — NLP Binary Classification]
* **Description:** Built and compared **supervised NLP models** for binary sentiment classification on the **IMDB movie review** dataset, establishing a strong baseline and an experimental sequential model. Evaluated both with standard classification metrics to analyze the tradeoff between simple averaged **embeddings and contextual sequence modeling**.
* **Key Achievements:** Processed **50K labeled reviews**(25K train / 25K test, perfectly balanced) with lowercasing, punctuation/stopword removal, lemmatization, and padding/truncation to max length 300, training embeddings from scratch. Built a MeanPool baseline (Embedding → Mean Pooling → Linear) that reached 87.3% accuracy, 0.8846 precision, 0.8584 recall, and 0.8713 F1, outperforming a BiLSTM experimental model (84.5% accuracy, 0.8426 F1).
Demonstrated that key sentiment-bearing words captured via averaged embeddings can beat heavier sequential models on IMDB, identifying overfitting and sequence-length challenges as the **BiLSTM's** main limitations.
* **Technologies:**  `Python`, `PyTorch`, `BiLSTM`, `AdamW`, `NLTK`, `Scikit-learn`

####  [CycleGAN Image Style Transfer - Monet]
* **Description:** Trained a **CycleGAN** to translate images between **Monet paintings and real photographs** in both directions without paired data, tackling a severe domain imbalance that caused discriminator collapse. Iterated across multiple training versions to stabilize learning and improve the Kaggle competition FID/MiFID score.
* **Key Achievements:** Implemented a ~60M-parameter **CycleGAN**(dual ResNet + self-attention generators ~27M each, PatchGAN + spectral-norm discriminators ~2.8M each) with adversarial (LSGAN), cycle-consistency, identity and VGG16 perceptual losses.
Diagnosed and addressed a 23:1 data imbalance (300 Monet vs. 7,038 photos) that collapsed the Monet discriminator (D_A ~0.034) by epoch 30, applying asymmetric augmentation, label smoothing and test-time augmentation across 3 training runs of 300 epochs each.
Demonstrated that key sentiment-bearing words captured via averaged embeddings can beat heavier sequential models on IMDB, identifying overfitting and sequence-length challenges as the BiLSTM's main limitations.
* **Technologies:**  `Python`, `PyTorch`, `CycleGAN`, `PatchGAN`, `Spectral Normalization`, `VGG16 Perceptual Loss`, `CUDA`

####  [Kayak Travel Platform](https://github.com/harshitasayala10/kayak-simulation)
* **Description:** A **full-stack travel booking platform** built on 7 Node.js microservices with **Kafka, Redis, and React on AWS EKS, featuring an AI recommendation agent** with natural-language intent parsing, deal detection, and **real-time WebSocket updates**. Load-tested with **JMeter** to deliver faster response times under peak traffic.
* **Key Achievements:** Architected a **full-stack booking platform with 7 Node.js microservices, Kafka event streams, Redis caching, and a React frontend on AWS EKS**.
Shipped an **AI recommendation agent** with natural-language intent parsing, deal detection, bundle suggestions, and WebSocket live updates.
Validated scalability with JMeter load tests, **achieving 3.8× throughput and 4.5× faster response times** under peak traffic.
* **Technologies:**  `FastAPI`, `Jmeter`, `LangChain`, `MySQL`, `Kafka`, `Docker`, `Python`, `Kubernetes`, `AWS`, `ReactJS`, `Redux`, `Node.js`, `Express`

####  [Reducing Attribute Leakage in Multi-Domain Text-to-Image Diffusion using LoRA and Attend-and-Excite]
* **Description:** Designed a **multi-domain text-to-image diffusion framework** to mitigate attribute leakage and improve compositional fidelity across diverse visual domains. Integrated domain-specific **LoRA adapters** (portrait, food, landscape) into **Stable Diffusion**, enabling controlled and realistic image generation from complex prompts.
* **Key Achievements:** Developed **semantic-weighted LoRA** composition using **CLIP** and **SBERT** embeddings to dynamically balance domain priors during inference.
Integrated the Attend-and-Excite (A&E) mechanism to enforce token-level attention consistency, significantly reducing cross-attribute interference in multi-object prompts.
Achieved notable performance gains over baseline SDXL models, improving FID (20.7), KID (0.0181), and CLIP-Sim (0.356), demonstrating enhanced prompt adherence and visual realism.
* **Technologies:** `Stable Diffusion`, `LoRA`, `CLIP`, `SBERT`, `PyTorch`, `Diffusion Models`

####  [Agentic Airbnb: AI-Integrated Travel Planning Platform Using LangChain & FastAPI](https://github.com/harshitasayala10/Agentic-airbnb)
* **Description:** Developed a full-stack, Airbnb-style travel planning platform with AI-powered recommendations, enabling users to search, book, and manage travel stays with personalized suggestions. Built end-to-end using ReactJS, Node.js (Express), MySQL, and Python FastAPI.
* **Key Achievements:** Designed and deployed 5 containerized microservices using Docker and Kubernetes, enabling seamless service orchestration and horizontal scalability on AWS.
Implemented Kafka message queues for asynchronous booking workflows, reducing booking confirmation latency by ~40% under simulated concurrent-user loads.
Integrated Redux for efficient frontend state management across sessions, listings, and bookings, improving UI data consistency and performance.
Leveraged LangChain-based AI pipelines to deliver intelligent travel recommendations and enhance user experience.
* **Technologies:**  `FastAPI`, `LangChain`, `MySQL`, `Kafka`, `Docker`, `Python`, `Kubernetes`, `AWS`, `ReactJS`, `Redux`, `Node.js`, `Express`

####  [Image Super-Resolution with Deep Learning]
* **Description:** Built a computationally efficient deep learning model that restores fine detail and removes noise from degraded low-resolution images, upscaling them to sharp high-resolution outputs. Optimized the model for real hardware deployment, balancing image quality against speed and compute cost on an NPU accelerator.
* **Key Achievements:** Developed a super-resolution model on the DIV2K, Flickr2K, and LSDIR datasets (88K+ training images), reconstructing 256×256 high-resolution images from inputs degraded with compression, Gaussian noise, and Gaussian blur, achieving 22 dB PSNR.
Optimized for efficiency under an NTIRE 2024-style scoring scheme weighting runtime (70%), FLOPs (15%), and parameters (15%), reaching 12 ms average runtime.
Benchmarked end-to-end on a Mobilint MLA100 NPU PCIe card using a trimmed-mean runtime across validation and test sets, earning a final score of 22.3.
* **Technologies:**  `Python`, `PyTorch`, `Deep Learning`, `CNN`, `Computer Vision`, `Super-Resolution`, `NumPy`, `OpenCV`, `NPU Deployment`

####  [Big Data Analytics for UK Police Crime Data](https://github.com/harshitasayala10/Big-Data-Analytics-for-UK-Police-Crime-Data)
* **Description:** Designed and implemented an end-to-end big data analytics pipeline using **Apache Hadoop (HDFS)** and **Apache Spark (PySpark, MLlib)** to analyze massive, inconsistent UK Police Crime datasets. Identified crime hotspots using K-Means clustering and revealed significant crime type-resolution relationships with Chi-Square Test ($p=0.0$).
* **Key Achievements:** Forecasted crime occurrences with **ARIMA(1,1,1)** (RMSE 16,830, MAE 13,426). Transformed data into actionable intelligence for data-informed policing.
* **Technologies:** `Apache Hadoop`, `Apache Spark (PySpark, MLlib)`, `ARIMA`, `K-Means`, `Chi-Square Test`, `Parquet`

####  [Spotify Trend and Popularity Prediction](https://github.com/harshitasayala10/DATA226/tree/main/Spotify_Project))
* **Description:** Designed a **real-time + batch data pipeline** using **Apache Airflow** to automate daily ingestion of 50K+ records from the Spotify API into **Snowflake**. Built and tuned **ARIMA-based time-series models** to forecast artist popularity.
* **Key Achievements:** Improved prediction accuracy by **30%** (RMSE reduced from 12.4 to 8.6). Developed interactive dashboards in **Apache Superset** for near-real-time trend insights.
* **Technologies:** `Apache Airflow`, `Snowflake`, `ARIMA`, `Apache Superset`, `Python`, `Pandas`, `Matplotlib`

####  [Air Quality Prediction and City Clustering Using Machine Learning for Sustainable Urban Planning](https://github.com/harshitasayala10/Air-Quality-Prediction-and-City-Clustering-Using-Machine-Learning-for-Sustainable-Urban-Planning)
* **Description:** Developed a comprehensive machine learning pipeline to **predict, classify, cluster, and forecast AQI** using diverse city-level datasets. Applied dual K-Means clustering to segment cities into 4 actionable urban typologies for targeted planning.
* **Key Achievements:** Achieved **94.10% R² and RMSE <6** for AQI prediction (Random Forest, XGBoost); improved classification accuracy to **89%** (SMOTE, PCA). Delivered insights supporting **United Nations Sustainable Development Goals(SDG 11)**.
* **Technologies:** `Random Forest`, `XGBoost`, `K-Means`, `SMOTE`, `PCA`, `Python`

####  [Cloud-Native Stock Price Analytics & Prediction](https://github.com/harshitasayala10/Cloud-Native-Stock-Price-Analytics-Prediction)
* **Description:** Developed a robust, **cloud-native data analytics pipeline** for real-time and historical stock market data, utilizing **Apache Airflow for orchestration, Snowflake as a data warehouse, dbt for transformation, and Preset for visualization.**
* **Key Achievements:** Automated data fetching (yfinance API), engineered streamlined ETL with **dbt** (including incremental UPSERT, error handling, dbt snapshots), and implemented **ARIMA models** for forecasting. Delivered actionable insights via intuitive Preset dashboards.
* **Technologies:** `Apache Airflow`, `Snowflake`, `dbt`, `Preset`, `ARIMA`, `yfinance API`, `SQL`

####  [Accident Hotspot Analysis]
* **Description:** Analyzed traffic accident records to pinpoint high-risk zones based on location, time, and driver behavior patterns, building a model that classified severe accidents with strong accuracy. Created interactive **Power BI dashboards** that highlight key risk factors to help city planners prioritize road safety and infrastructure improvements.
* **Key Achievements:** Analyzed **50K+ traffic** accident records with **Python (Pandas), K-Means clustering, and XGBoost** to identify high-risk zones from spatial, temporal, and behavioral patterns.
Engineered **geospatial and time-based** features to model accident severity, achieving **95.88% accuracy with F1 of 0.83** on severe-accident classification.
Built interactive **Power BI dashboards surfacing cluster-level risk factors** to support municipal planning and infrastructure prioritization decisions.
* **Technologies:** `PowerBI`, `Python`, `Pandas`, `XGBoost`, `K-means`, `Scikit learn`

---

### Let's Connect!

I'm always open to discussing **collaborations, opportunities, and innovative data solutions**. Feel free to connect or reach out!

* **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/harshita-sayala-84a9a9204/)
* **Email:** [Gmail](harshita.sayala@sjsu.edu)
---

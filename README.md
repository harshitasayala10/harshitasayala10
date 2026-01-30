# Hi, I'm Harshita Sayala! 👋

### 🎓 About Me

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

### 🛠️ Technical Skills

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

### 🚀 My Featured Projects

Here are some of my key projects, demonstrating my expertise in data analytics, machine learning, and data engineering. Each link leads to a dedicated repository with full details, code, and often visualizations or demos.

#### ✈️ [AI-Integrated Travel Planning Platform Using LangChain & FastAPI](https://github.com/harshitasayala10/Agentic-airbnb)
* **Description:** Developed a full-stack, Airbnb-style travel planning platform with AI-powered recommendations, enabling users to search, book, and manage travel stays with personalized suggestions. Built end-to-end using ReactJS, Node.js (Express), MySQL, and Python FastAPI.
* **Key Achievements:** Designed and deployed 5 containerized microservices using Docker and Kubernetes, enabling seamless service orchestration and horizontal scalability on AWS.
Implemented Kafka message queues for asynchronous booking workflows, reducing booking confirmation latency by ~40% under simulated concurrent-user loads.
Integrated Redux for efficient frontend state management across sessions, listings, and bookings, improving UI data consistency and performance.
Leveraged LangChain-based AI pipelines to deliver intelligent travel recommendations and enhance user experience.
* **Technologies:**  `FastAPI`, `LangChain`, `MySQL`, `Kafka`, `Docker`, `Python`, `Kubernetes`, `AWS`, `ReactJS`, `Redux`, `Node.js`, `Express`

#### 🎨 [Reducing Attribute Leakage in Multi-Domain Text-to-Image Diffusion using LoRA and Attend-and-Excite]
* **Description:** Designed a **multi-domain text-to-image diffusion framework** to mitigate attribute leakage and improve compositional fidelity across diverse visual domains. Integrated domain-specific **LoRA adapters** (portrait, food, landscape) into **Stable Diffusion**, enabling controlled and realistic image generation from complex prompts.
* **Key Achievements:** Developed **semantic-weighted LoRA** composition using **CLIP** and **SBERT** embeddings to dynamically balance domain priors during inference.
Integrated the Attend-and-Excite (A&E) mechanism to enforce token-level attention consistency, significantly reducing cross-attribute interference in multi-object prompts.
Achieved notable performance gains over baseline SDXL models, improving FID (20.7), KID (0.0181), and CLIP-Sim (0.356), demonstrating enhanced prompt adherence and visual realism.
* **Technologies:** `Stable Diffusion`, `LoRA`, `CLIP`, `SBERT`, `PyTorch`, `Diffusion Models`

#### 🚨 [Big Data Analytics for UK Police Crime Data](https://github.com/harshitasayala10/Big-Data-Analytics-for-UK-Police-Crime-Data)
* **Description:** Designed and implemented an end-to-end big data analytics pipeline using **Apache Hadoop (HDFS)** and **Apache Spark (PySpark, MLlib)** to analyze massive, inconsistent UK Police Crime datasets. Identified crime hotspots using K-Means clustering and revealed significant crime type-resolution relationships with Chi-Square Test ($p=0.0$).
* **Key Achievements:** Forecasted crime occurrences with **ARIMA(1,1,1)** (RMSE 16,830, MAE 13,426). Transformed data into actionable intelligence for data-informed policing.
* **Technologies:** `Apache Hadoop`, `Apache Spark (PySpark, MLlib)`, `ARIMA`, `K-Means`, `Chi-Square Test`, `Parquet`

#### 🏙️ [Air Quality Prediction and City Clustering Using Machine Learning for Sustainable Urban Planning](https://github.com/harshitasayala10/Air-Quality-Prediction-and-City-Clustering-Using-Machine-Learning-for-Sustainable-Urban-Planning)
* **Description:** Developed a comprehensive machine learning pipeline to **predict, classify, cluster, and forecast AQI** using diverse city-level datasets. Applied dual K-Means clustering to segment cities into 4 actionable urban typologies for targeted planning.
* **Key Achievements:** Achieved **94.10% R² and RMSE <6** for AQI prediction (Random Forest, XGBoost); improved classification accuracy to **89%** (SMOTE, PCA). Delivered insights supporting **United Nations Sustainable Development Goals(SDG 11)**.
* **Technologies:** `Random Forest`, `XGBoost`, `K-Means`, `SMOTE`, `PCA`, `Python`

#### 🎵 [Spotify Trend and Popularity Prediction](https://github.com/wcc0/DATA226)
* **Description:** Designed a **real-time + batch data pipeline** using **Apache Airflow** to automate daily ingestion of 50K+ records from the Spotify API into **Snowflake**. Built and tuned **ARIMA-based time-series models** to forecast artist popularity.
* **Key Achievements:** Improved prediction accuracy by **30%** (RMSE reduced from 12.4 to 8.6). Developed interactive dashboards in **Apache Superset** for near-real-time trend insights.
* **Technologies:** `Apache Airflow`, `Snowflake`, `ARIMA`, `Apache Superset`, `Python`, `Pandas`, `Matplotlib`

#### 💹 [Cloud-Native Stock Price Analytics & Prediction](https://github.com/harshitasayala10/Cloud-Native-Stock-Price-Analytics-Prediction)
* **Description:** Developed a robust, **cloud-native data analytics pipeline** for real-time and historical stock market data, utilizing **Apache Airflow for orchestration, Snowflake as a data warehouse, dbt for transformation, and Preset for visualization.**
* **Key Achievements:** Automated data fetching (yfinance API), engineered streamlined ETL with **dbt** (including incremental UPSERT, error handling, dbt snapshots), and implemented **ARIMA models** for forecasting. Delivered actionable insights via intuitive Preset dashboards.
* **Technologies:** `Apache Airflow`, `Snowflake`, `dbt`, `Preset`, `ARIMA`, `yfinance API`, `SQL`

#### ⚖️ LegalDoc AI – Intelligent Legal Document Assistant using LLMs and RAG
* **Description:** Built a **domain-specific chatbot** using **Retrieval-Augmented Generation (RAG)** to assist lawyers in legal research. Processed and embedded 10,000+ legal document chunks for semantic search.
* **Key Achievements:** Integrated **OpenAI GPT-3.5** for clause-level answers with **>90% answer relevance**. Created an interactive **Streamlit** interface, reducing review time by ~60 mins/doc. Orchestrated pipeline with **LangChain**.
* **Technologies:** `LLMs (OpenAI GPT-3.5)`, `RAG`, `LangChain`, `Streamlit`, `SentenceTransformers`, `FAISS`, `Python`

---

### Let's Connect!

I'm always open to discussing **collaborations, opportunities, and innovative data solutions**. Feel free to connect or reach out!

* **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/harshita-sayala-84a9a9204/)
* **Email:** [Gmail](harshita.sayala@sjsu.edu)
---

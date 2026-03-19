[Your Full Name]
[Your Address]
[City, Postal Code, Ireland]
[Email Address] | [Phone Number]

19 March 2026

Dr Ken Bruton
Intelligent Energy Research Group (IERG)
School of Engineering and Architecture
University College Cork

**Subject: Application for PhD Studentship — AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance (FLARE)**

Dear Dr Bruton,

I am writing to apply for the PhD studentship in AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance within the FLARE project at University College Cork. I am currently completing an MSc in Software Design with AI at TUS Athlone (results expected May–August 2026), and I am eager to transition into doctoral research in industrial AI — an area where I believe rigorous machine learning methodology can have direct, measurable impact on industrial reliability and sustainability.

To demonstrate the depth and relevance of my preparation, I developed **SENTINEL**, a full-stack industrial AI platform for predictive maintenance of turbofan engines, built on the NASA C-MAPSS benchmark dataset. This project goes beyond a standard coursework pipeline: it is a production-grade research system with a FastAPI backend, real-time Next.js dashboard, Docker deployment, and a three-phase ML pipeline covering anomaly detection, RUL estimation, and explainability.

The key research findings from this work are directly relevant to the FLARE PhD topic:

**1. Operating Condition Normalisation for multi-condition fault detection.** I identified that LSTM autoencoders trained on multi-condition datasets (FD002, FD004) produce near-zero anomaly F1 scores not due to model failure, but because operating regime shifts are confounded with degradation signals. I designed and validated a KMeans regime clustering normalisation approach that resolves this, improving FD002 anomaly F1 from 0.06 to 0.61 (a tenfold improvement) and FD004 from 0.02 to 0.45. This finding has direct implications for industrial deployment where operating conditions vary continuously.

**2. Monotonic RUL enforcement for physically consistent predictions.** XGBoost RUL predictions violate the physical constraint that remaining life must be non-increasing. I applied per-engine isotonic regression post-processing to enforce this, reducing RMSE by 8.4% on FD002 and 12.8% on FD004 without retraining. This kind of domain-knowledge integration into ML systems is, I believe, central to trustworthy industrial AI.

**3. Cross-method explainability validation.** I implemented both SHAP and LIME attribution and measured their agreement — an approach that provides a self-consistency check on feature attribution fidelity. SHAP and LIME agreed on top-5 features in over 80% of test instances, with divergence at near-failure cycles flagged as a diagnostic signal.

Beyond the ML results, SENTINEL is built as an enterprise platform: a FastAPI backend with JWT authentication, asynchronous SQLAlchemy database, Redis pub/sub, and Celery workers for asynchronous inference; a Next.js frontend with real-time WebSocket updates, animated fleet dashboards, per-asset trend charts, and a SHAP bar chart visualisation; and a full CI pipeline via GitHub Actions. This full-stack approach reflects my belief that research AI must be deployable to be useful, and it demonstrates the software engineering capability required to deliver on an applied industrial research programme such as FLARE.

My MSc programme at TUS Athlone covered machine learning, deep learning, NLP, and data engineering, and I have developed strong Python and ML engineering skills through both coursework and independent project work. I am comfortable working with PyTorch, scikit-learn, XGBoost, and the Python scientific stack, and I have experience designing and validating reproducible, config-driven experimental pipelines.

I am motivated by the FLARE project specifically because it targets the intersection of AI research and real industrial energy systems — an area where robust, interpretable, and deployable methods matter far more than benchmark-optimised models. I am confident that my combination of ML research ability and full-stack engineering experience would allow me to contribute productively to your research group from an early stage.

I would very much welcome the opportunity to discuss my application further. The SENTINEL codebase and full benchmark results are available at https://github.com/AhmadKhan46/sentinel-ai for your review. I have attached my CV, academic transcripts, and referee contact details.

Thank you for your time and consideration.

Yours sincerely,
[Your Full Name]

---

**Attachments:**
- CV
- Academic transcripts (TUS Athlone MSc Software Design with AI — in progress)
- Referee contact details (x2)
- Supporting technical brief: `reports/application/project_evidence_ucc.pdf`
- GitHub repository: https://github.com/AhmadKhan46/sentinel-ai (includes live benchmark results, RESEARCH_FINDINGS.md, full codebase)

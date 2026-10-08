# PTP Credibility Engine

A machine learning system that estimates whether a Promise to Pay (PTP) made during a debt-collection interaction is likely to be credible and achievable.

A borrower agreeing to pay does not necessarily mean the promise will be kept. A promise can be genuine but financially infeasible, vague, made under pressure, or inconsistent with previous payment behaviour.

This project explores how conversation-level signals and borrower history can be combined to estimate PTP credibility and classify the type of promise being made.

## Live Demo

**[Try the deployed Streamlit app](https://ptp-credibility-4wxlf26uhebhgpzam53mqz.streamlit.app/)**

The app allows users to enter borrower information and a collection-call transcript and receive a predicted PTP credibility score, PTP type, and supporting signals.

## What the model does

The system produces two main outputs:

* **PTP credibility** — an estimate of how likely the promised payment is to be fulfilled.
* **PTP type** — a classification of what kind of promise the interaction represents.

The current prototype supports categories such as:

* Genuine and feasible
* Genuine but not feasible
* Escape promise
* Third-party promise
* Repeat promiser
* Agent-recorded or agent-pushed PTP

The goal is not simply to label a promise as "good" or "bad", but to understand *why* a promise may be weak and what that could mean for the next collection action.

## How it works

The overall pipeline is:

```text
Synthetic Data
      ↓
Feature Preparation
      ↓
Model Training
      ↓
Trained Model
      ↓
Prediction Pipeline
      ↓
Streamlit Application
      ↓
PTP Credibility + PTP Type
```

For a collection-call workflow, the intended flow is:

```text
Call / Transcript
       ↓
Relevant PTP Signals
       ↓
Credibility Model
       ↓
Credibility Score
       ↓
PTP Type
       ↓
Suggested Follow-up
```

For example, if a borrower agrees to pay after receiving their salary but chooses a date before the expected salary date, the system should be able to distinguish a potentially genuine but infeasible promise from other types of weak commitments.

## Signals used

The model can use signals such as:

* Who proposed the payment date
* Whether a specific payment amount was agreed upon
* Previous PTP keep rate
* Previous payment behaviour
* Promised date relative to expected cash flow
* Number of previously broken promises
* Whether negotiation occurred during the call
* Conditional language such as "salary aayi toh" or "try karunga"
* Whether the borrower gave a clear commitment or a vague response
* Payment-link activity after the call

These signals are intended to provide context rather than act as standalone evidence of borrower behaviour.

## What I built

The repository contains the complete prototype pipeline, from synthetic data generation and model training to a deployed prediction interface.

| File                         | Purpose                                             |
| ---------------------------- | --------------------------------------------------- |
| `generate_synthetic_data.py` | Generates synthetic borrower and PTP data           |
| `ptp_synthetic.csv`          | Synthetic dataset used for development and training |
| `train_model.py`             | Trains and saves the machine learning model         |
| `ptp_pipeline.py`            | Feature preparation and prediction pipeline         |
| `ptp_engine.joblib`          | Trained model used by the application               |
| `app.py`                     | Streamlit interface for interactive predictions     |
| `livecall.py`                | Live-call interface prototype                       |
| `requirements.txt`           | Python dependencies                                 |

## Why synthetic data?

Real borrower and collection data is sensitive and is not appropriate for a public portfolio project.

Synthetic data makes it possible to build and demonstrate the complete ML workflow without exposing customer information.

The trade-off is important: performance on synthetic data does **not** demonstrate that the model will perform similarly on real borrowers.

A real deployment would require:

* Properly labelled historical outcomes
* A precise definition of a kept PTP
* Representative conversation and transcript data
* Feature validation with domain experts
* Probability calibration
* Out-of-time validation
* Monitoring for model drift
* Evaluation across different portfolios and borrower segments

## A problem with historical data

Collection data also has an important limitation.

Historical data records outcomes under the actions that were already taken. It does not directly tell us what would have happened if a different intervention had been used.

For example:

```text
Historical system
      ↓
  Action taken
      ↓
Borrower outcome
```

A model may predict which PTPs are likely to be broken, but that does not automatically prove that changing the collection strategy will improve recovery.

A production system would therefore need controlled experiments or other causal evaluation methods to measure whether model-driven interventions actually improve outcomes.

## Responsible use

A low credibility score should not automatically result in harsher treatment.

A borrower may genuinely intend to pay but be unable to meet the proposed date because of their financial situation. That is different from making a promise simply to end a conversation.

A production system should therefore consider:

* Why a PTP received a low score
* Whether hardship indicators are present
* Whether the prediction is reliable for the relevant portfolio
* Whether the model behaves differently across borrower groups
* Whether predictions can be explained to collection agents
* How customer and conversation data is stored and retained

The model should support human decision-making rather than replace it.

## Current status

The prototype is currently deployed as a Streamlit web application.

The project includes:

* Synthetic data generation
* Feature engineering
* Machine learning model training
* Saved prediction model
* Prediction pipeline
* Interactive Streamlit interface
* Public deployment

The current model is a prototype trained on synthetic data and should not be interpreted as a production-ready collections model.

## Running locally

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to regenerate the synthetic dataset:

```bash
python generate_synthetic_data.py
```

Retrain the model:

```bash
python train_model.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Tech stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Machine Learning
* Synthetic data generation

## Limitations

This is a portfolio prototype rather than a production lending or collections system.

The main limitation is the use of synthetic data. The model has not been validated against a real collection portfolio, so its predictions should not be interpreted as real-world performance.

The live-call functionality is also a prototype and would require additional work for production deployment, including reliable speech/transcript processing, latency handling, monitoring, security, and integration with collection systems.

## Future work

Potential next steps include:

```text
Synthetic Prototype
       ↓
Real / Representative Data
       ↓
Model Validation
       ↓
Model Explanation
       ↓
Real-time Transcript Processing
       ↓
Controlled Intervention Testing
       ↓
Production Monitoring
```

## Author

**Vatsal Patel**

B.Tech, Civil Engineering
IIT Kharagpur

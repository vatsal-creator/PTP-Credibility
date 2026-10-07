# PTP Credibility Engine

A machine learning project for detecting whether a Promise to Pay (PTP) is likely to be genuine and achievable.

In debt collection calls, a borrower may agree to make a payment on a particular date. But not every promise has the same meaning. Some borrowers genuinely intend to pay, some agree to a date that doesn't match their cash flow, and some may simply agree to end the conversation.

The idea behind this project is to use information around the conversation and the borrower's history to estimate how credible a PTP is.

## What the model does

The system is designed to estimate:

* PTP credibility — how likely the promised payment is to actually happen.
* PTP type — what kind of promise the conversation appears to represent.

Possible PTP types include:

* Genuine and feasible
* Genuine but not feasible
* Escape promise
* Third-party promise
* Repeat promiser
* Agent-recorded or agent-pushed PTP

The distinction matters because the next action should depend on why a promise looks weak, rather than simply treating every low-confidence PTP the same way.

## The basic idea

The current pipeline looks like this:

```
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
PTP Credibility + PTP Type
```

The eventual system could work during a live collection call:

```
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

For example, if a borrower agrees to pay after their salary date but the proposed PTP date is earlier, the system should be able to distinguish a potentially genuine but infeasible promise from an intentionally misleading one.

## Signals

The model can use signals such as:

* Whether the borrower or agent proposed the payment date
* Whether a specific payment amount was agreed upon
* Previous PTP keep rate
* Previous payment behaviour
* Promised date compared with expected cash flow
* Number of previously broken promises
* Whether there was negotiation during the call
* Conditional language such as "salary aayi toh" or "try karunga"
* Whether the borrower gave a clear commitment or a vague response
* Payment-link activity after the call

For a real system, these signals would need to be validated carefully. A single phrase or behaviour should not be treated as proof that a borrower is being dishonest.

## What I built

The current repository contains the core machine learning pipeline:

| File                         | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `generate_synthetic_data.py` | Generates synthetic data for development |
| `ptp_synthetic.csv`          | Synthetic dataset used by the model      |
| `train_model.py`             | Trains the machine learning model        |
| `ptp_engine.joblib`          | Saved trained model                      |
| `ptp_pipeline.py`            | Loads the model and performs predictions |

The current version uses synthetic data, so the model should be treated as a prototype rather than a validated real-world collection model.

## Why synthetic data?

Real borrower and collection data is sensitive and cannot simply be used for a portfolio project.

Synthetic data allows the complete pipeline to be developed and tested without exposing real customer information.

However, good performance on synthetic data does not mean that the model will perform equally well on real borrowers.

A real implementation would require:

* Properly labelled historical outcomes
* A clear definition of what counts as a "kept" PTP
* Real conversation/transcript data
* Careful feature validation
* Calibration testing
* Out-of-time validation
* Monitoring for model drift

## A problem with historical data

There is another important issue with this type of model.

Historical collection data records what happened under the decisions that were already made. It does not show what would have happened if a different action had been taken.

For example:

```
Historical system
      ↓
  Action taken
      ↓
Borrower outcome
```

This makes it difficult to determine whether a model-driven intervention actually improves recovery.

A stronger production evaluation would therefore need controlled testing to compare different interventions rather than relying only on offline model accuracy.

## Responsible use

A low credibility score should not automatically mean harsher treatment.

Someone may genuinely want to pay but be unable to meet the proposed date because of financial circumstances. That is different from intentionally making a promise just to end a call.

A production system should therefore consider:

* Why a PTP received a low score
* Whether hardship indicators are present
* Whether the prediction is reliable for the relevant portfolio
* Whether the model is becoming biased toward particular borrower groups
* Whether predictions can be explained to agents and supervisors
* How customer and conversation data is stored and retained

The model should support better decisions, not replace human judgement.

## Current status

The core ML pipeline is built.

The repository currently contains the synthetic dataset, training script, trained model and prediction pipeline.

The next step is to put a simple interface around the model so that a user can enter PTP information and receive:

```
PTP Credibility Score
        +
Predicted PTP Type
        +
Relevant reasoning/signals
```

The longer-term goal is to turn the pipeline into a small working application that demonstrates the complete flow from input → model → prediction.

## Project structure

```
PTP-Credibility/
│
├── generate_synthetic_data.py
├── ptp_synthetic.csv
├── train_model.py
├── ptp_pipeline.py
├── ptp_engine.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

## Running locally

Create a virtual environment:

```
python -m venv venv
```

Activate it on Windows:

```
venv\Scripts\activate
```

Install the required packages:

```
pip install -r requirements.txt
```

Generate the synthetic dataset:

```
python generate_synthetic_data.py
```

Train the model:

```
python train_model.py
```

Run the prediction pipeline:

```
python ptp_pipeline.py
```

## Tech stack

* Python
* Pandas
* Scikit-learn
* Joblib
* Machine Learning
* Synthetic data generation

## Limitations

This is currently a prototype.

The biggest limitation is the use of synthetic data. The model has not been validated against a real collection portfolio, so the reported predictions should not be interpreted as real-world performance.

The live-call component is also not implemented yet. The current project focuses on the underlying ML pipeline.

## Next steps

The planned development path is:

```
Core ML Pipeline
      ↓
Interactive Interface
      ↓
Model Explanation
      ↓
Real-time prediction flow
      ↓
Deployment
```

The immediate goal is to turn the existing Python pipeline into a usable application rather than keeping it as a collection of scripts.

## Author

**Vatsal Patel**

B.Tech, Civil Engineering
IIT Kharagpur

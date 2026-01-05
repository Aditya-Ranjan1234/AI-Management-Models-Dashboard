# AI Management Models Dashboard

This project implements three AI models to support managerial transformation in hybrid work environments.

## Models

1.  **Productivity Predictor (RQ1)**: Analyzes performance in distributed work.
    *   *Folder*: `Productivity_Predictor`
    *   *Goal*: Predict productivity levels based on remote work habits.
2.  **Attrition Risk Model (RQ2)**: Addresses resilience & cohesion.
    *   *Folder*: `Attrition_Risk_Model`
    *   *Goal*: Predict employee attrition risk.
3.  **Task Recommender (RQ3)**: Implements the "Human-AI Collaboration" framework.
    *   *Folder*: `Task_Recommender`
    *   *Goal*: Classify tasks as suitable for AI automation, Augmentation, or Human execution.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data & Train Models**:
    Run the provided scripts to generate synthetic data (mimicking the target Kaggle datasets) and train the models locally.

3.  **Run Dashboard**:
    ```bash
    streamlit run app.py
    ```

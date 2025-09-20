# Telco-Customer-Classification-Model


``` mermaid

graph TD
    A[Start] --> B{Save Files};
    B --> B1[app.py];
    B --> B2[README.md];
    B --> B3[AdaBoost_best_model.pkl];
    B --> C{Create requirements.txt};
    C --> D{Add Dependencies to requirements.txt};
    D --> E[streamlit, pandas, numpy, scikit-learn];
    E --> F{Install Dependencies via Terminal};
    F --> G["pip install -r requirements.txt"];
    G --> H{Run the App};
    H --> I["streamlit run app.py"];
    I --> J[View App in Browser];
    J --> K[End];

    style A fill:#e94560,stroke:#c73049,stroke-width:2px,color:#fff
    style K fill:#e94560,stroke:#c73049,stroke-width:2px,color:#fff
    style B fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#e0e0e0
    style C fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#e0e0e0
    style D fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#e0e0e0
    style F fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#e0e0e0
    style H fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#e0e0e0
    style J fill:#4dff91,stroke:#39e67a,stroke-width:2px,color:#1a1a2e
    style G fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
    style I fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff


```

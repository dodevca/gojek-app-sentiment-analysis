# Gojek App Sentiment Analysis

## About This Project
A sentiment analysis project on user reviews of the Gojek mobile application (version 5.18.2) from Google Play Store. This project combines data scraping, preprocessing, machine learning modeling, and web-based visualization to deliver actionable insights about user feedback. The application features an interactive dashboard for sentiment summary and detailed analysis.

## Tech Stack & Tools
| Category     | Stack                        |
|--------------|------------------------------|
| Data Analysis| Python (Pandas, NumPy)        |
| Machine Learning | Scikit-learn (TF-IDF, Classifiers), NLTK (optional) |
| Visualization | Matplotlib, Seaborn          |
| Web Interface | Flask, Jinja2 Templates, Bootstrap |
| Deployment    | Streamlit (Optional for Demo)|
| Dataset Source| Google Play Reviews (Gojek v5.18.2) |

## Key Features
- Google Play review scraping & preprocessing (cleaning, tokenizing).
- Sentiment classification (positive, neutral, negative) using supervised learning models.
- Visualization of sentiment distribution and keyword frequency.
- Web-based dashboard for interactive exploration of analysis results.
- API endpoint integration for future scalability.

## Live Demo
You can access the deployed web application at:
[https://gojek-sentiment.dodevca.com](https://gojek-sentiment.dodevca.com)

## Installation and Usage (Local Setup)
1. Clone this repository.
2. Create and activate a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    # OR
    venv\Scripts\activate  # For Windows
    ```
3. Install required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook for data preprocessing & model training.
    ```bash
    jupyter notebook NLP.ipynb
    ```
5. To run the web application (Flask):
    ```bash
    python app.py
    ```
6. Open your browser and go to `http://localhost:5000`.

## Future Improvements
- Integrate more advanced models (BERT, LSTM) for enhanced prediction accuracy.
- Add real-time review fetching using Google Play API/Scraper.
- Expand dataset to include version-wise sentiment comparison.
- Deploy as a scalable microservice API for sentiment analysis.

## Contributors
This project is collaboratively developed by:
- [Dodevca](https://github.com/dodevca)
- [Illyaz Arya](https://github.com/McIllyaz)
- [Khayatttt](https://github.com/Khayatttt)

## Contact & Collaboration
Interested in collaborating or enhancing this project?
Reach me at [LinkedIn](https://linkedin.com/in/dodevca) or visit [dodevca.com](https://dodevca.com).

## Signature
Initiated by **Dodevca & Team**, open for collaboration and continuous refinement.
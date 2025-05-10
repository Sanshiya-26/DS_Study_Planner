def safe_int(value, fallback):
    try:
        return int(value)
    except (ValueError, TypeError):
        return fallback

def constrain(value, low, high):
    return max(low, min(value, high))

def generate_plan(user_input):
    goal = user_input.get("goal", "Your Goal")

    # Clean, limited input
    hours_per_day = constrain(safe_int(user_input.get("hours_per_day"), 1), 1, 15)
    days_per_week = constrain(safe_int(user_input.get("days_per_week"), 1), 1, 7)
    preferred_style = user_input.get("preferred_style", "").lower()
    if preferred_style not in ["article", "practice"]:
        preferred_style = "article"

    print("DEBUG INPUT:", goal, hours_per_day, days_per_week, preferred_style)

    # Quotes for motivation
    quotes = [
        "Push harder than yesterday.",
        "Never ever give up.",
        "Pain today, strength tomorrow.",
        "Keep going, you're doing great.",
        "Believe in yourself.",
        "Stay focused and never give up.",
        "Your limitation is your imagination.",
        "Get out of your comfort zone.",
        "Stay consistent, stay motivated!",
        "End the day with satisfaction."
    ]

    # Topics for each day (100 topics)
    topics = [
        "What is Data Science? Overview & Applications",
        "Introduction to Python Programming",
        "Python Data Types and Variables",
        "Conditional Statements and Loops in Python",
        "Functions and Lambda Expressions in Python",
        "Working with Lists, Tuples, and Dictionaries",
        "File Handling in Python",
        "Introduction to NumPy: Arrays and Matrices",
        "NumPy Array Operations",
        "NumPy Mathematical Functions",
        "Introduction to Pandas: DataFrames and Series",
        "Data Cleaning with Pandas",
        "Data Manipulation with Pandas",
        "Data Aggregation and Grouping in Pandas",
        "Introduction to Data Visualization",
        "Basic Plotting with Matplotlib",
        "Advanced Plotting Techniques in Matplotlib",
        "Introduction to Seaborn",
        "Statistical Plots with Seaborn",
        "Data Visualization Best Practices",
        "Creating Interactive Visualizations",
        "Exploratory Data Analysis (EDA) Techniques",
        "Handling Missing Data",
        "Data Standardization and Normalization",
        "Data Transformation Techniques",
        "Introduction to Statistics for Data Science",
        "Measures of Central Tendency",
        "Measures of Dispersion (Variance, Standard Deviation)",
        "Probability Fundamentals",
        "Probability Distributions (Normal, Binomial)",
        "Hypothesis Testing Basics",
        "T-Tests and Z-Tests",
        "Chi-Square Test",
        "Analysis of Variance (ANOVA)",
        "Correlation and Covariance",
        "Introduction to Regression Analysis",
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression",
        "Logistic Regression",
        "Introduction to Machine Learning",
        "Types of Machine Learning (Supervised, Unsupervised, Reinforcement)",
        "Data Preprocessing Techniques for ML",
        "Feature Selection and Extraction",
        "Introduction to Scikit-Learn",
        "Implementing Linear Regression with Scikit-Learn",
        "Implementing Logistic Regression with Scikit-Learn",
        "K-Nearest Neighbors Algorithm",
        "Support Vector Machines (SVM)",
        "Decision Trees and Random Forests",
        "Gradient Boosting Algorithms (XGBoost, LightGBM)",
        "Introduction to Clustering Algorithms",
        "K-Means Clustering",
        "Hierarchical Clustering",
        "DBSCAN Clustering",
        "Principal Component Analysis (PCA)",
        "Association Rule Learning (Apriori, FP-Growth)",
        "Implementing Recommender Systems",
        "Introduction to Natural Language Processing (NLP)",
        "Text Preprocessing Techniques (Tokenization, Lemmatization)",
        "Sentiment Analysis",
        "Named Entity Recognition (NER)",
        "Introduction to Deep Learning",
        "Neural Networks Basics",
        "Activation Functions (ReLU, Sigmoid)",
        "Backpropagation and Gradient Descent",
        "Implementing Neural Networks with TensorFlow",
        "Implementing Neural Networks with PyTorch",
        "Convolutional Neural Networks (CNNs)",
        "Implementing CNNs with TensorFlow",
        "Recurrent Neural Networks (RNNs)",
        "Long Short-Term Memory (LSTM) Networks",
        "Implementing LSTM with TensorFlow",
        "Transfer Learning Concepts",
        "Implementing Transfer Learning",
        "Generative Adversarial Networks (GANs)",
        "Implementing GANs with PyTorch",
        "Reinforcement Learning Basics",
        "Q-Learning Algorithm",
        "Markov Decision Process (MDP)",
        "Implementing Q-Learning in Python",
        "Introduction to Data Engineering",
        "Working with SQL for Data Analysis",
        "Data Extraction Techniques",
        "Data Cleaning and Transformation Techniques",
        "Data Warehousing Basics",
        "Introduction to Big Data (Hadoop, Spark)",
        "Working with PySpark",
        "Introduction to Data Pipelines",
        "Implementing Data Pipelines with Airflow",
        "Introduction to Model Evaluation",
        "Cross-Validation Techniques",
        "Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)",
        "Model Deployment Basics",
        "Deploying Models with Flask",
        "Deploying Models with Streamlit",
        "Model Monitoring and Maintenance",
        "Building Data Science Projects",
        "Creating a Data Science Portfolio",
        "Data Science Career Planning and Interview Preparation"
    ]

    total_days = 100
    plan = []

    for day in range(1, total_days + 1):
        # Assign quote based on day
        quote = quotes[(day - 1) % len(quotes)]
        topic = topics[day - 1]

        # Highlight every 10th day
        if day % 10 == 0:
            plan.append(f"<b style='font-size: 1.5em; color: purple;'>Day {day}: {quote}</b><br>Topic: {topic}")
        else:
            # Regular day with quote and topic
            plan.append(f"Day {day}: {quote}<br>Topic: {topic}")

    print("Generated Plan Sample:", plan[:5])

    return plan, topics

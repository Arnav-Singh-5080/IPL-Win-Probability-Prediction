# 🏏 IPL Win Probability Prediction — Machine Learning Project

A machine learning project that predicts **win probabilities of IPL teams** during a live chase, using real match data and ball-by-ball statistics.  
This notebook implements a complete ML workflow — including **data cleaning, merging, feature engineering, training, and prediction visualization** — built using **Pandas, NumPy, Matplotlib, and Scikit-learn**.

---

## 📘 1. Project Overview

The goal of this project is to predict the **probability of a team winning** an IPL match while chasing, based on match progression and performance metrics such as **runs left**, **balls left**, **wickets remaining**, and **run rates**.

The project demonstrates how real-world sports data can be transformed into a **binary classification problem**, where the model outputs the chances of the batting team **winning (1)** or **losing (0)**.

---

## 🎯 2. Objectives

* Analyze and preprocess IPL datasets (`matches.csv` and `deliveries.csv`).
* Engineer real-time match features such as **CRR**, **RRR**, and **runs_left**.
* Build a predictive ML model to estimate winning probabilities.
* Implement a function to track and visualize **match progression** over overs.
* Present a clean and interpretable pipeline for sports analytics.

---

## 🧩 3. Dataset Description

**Source:** [Kaggle IPL Dataset](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)

| File             | Description                                                        |
| :--------------- | :----------------------------------------------------------------- |
| `matches.csv`    | Contains high-level match details such as teams, city, and winner. |
| `deliveries.csv` | Contains ball-by-ball data for each match.                         |

**Merged Dataset Includes:**

| Column       | Description                    |
| :----------- | :----------------------------- |
| match_id     | Unique match identifier        |
| batting_team | Team currently batting         |
| bowling_team | Team currently bowling         |
| city         | Location of the match          |
| runs_left    | Runs required to win           |
| balls_left   | Balls remaining in the innings |
| wickets      | Wickets remaining              |
| total_runs_x | Target score                   |
| crr          | Current Run Rate               |
| rrr          | Required Run Rate              |

---

## ⚙️ 4. Workflow Summary

### 🧹 Step 1: Data Preprocessing

* Loaded and explored `matches.csv` and `deliveries.csv` using **Pandas**.  
* Merged datasets using `match_id`.  
* Dropped unnecessary columns and handled missing values.  
* Created filtered datasets for innings, teams, and valid match states.

### 🧮 Step 2: Feature Engineering

Added custom features essential for match prediction:

* `runs_left = target - total_runs_y`
* `balls_left = 120 - (over * 6 + ball)`
* `wickets = 10 - cumulative(player_dismissed)`
* `crr = current_score / overs_played`
* `rrr = (runs_left * 6) / balls_left`

These features represent **real-time match context**.

### 🤖 Step 3: Model Building

* Split the processed dataset into training and testing sets using `train_test_split()`.  
* Trained a **Logistic Regression model** using **Scikit-learn**.  
* Used a `Pipeline` to handle encoding and modeling in one step.  
* Evaluated model performance using accuracy and prediction probabilities.

### 📈 Step 4: Match Progression Function

Defined the function:

`match_progression(x_df, match_id, pipe)`

This function:

* Filters deliveries at the **end of each over**.  
* Computes **win** and **lose probabilities** for each over using the trained model.  
* Calculates **runs scored** and **wickets lost per over**.  
* Returns a DataFrame summarizing the match progression.

### 📊 Step 5: Visualization

Used **Matplotlib** to visualize:

* Win and loss probabilities after each over.  
* How momentum shifted as the match progressed.

---

## 🧾 5. Example Output

| Over | Runs Scored | Wickets | Win % | Lose % |
| :--- | :---------- | :------ | :---- | :----- |
| 1    | 8           | 0       | 45.8  | 54.2   |
| 5    | 49          | 1       | 62.3  | 37.7   |
| 10   | 88          | 3       | 71.5  | 28.5   |
| 15   | 123         | 5       | 54.2  | 45.8   |

**Visualization Example:**  
📈 *Win probability curve* showing team momentum throughout the chase.

---

## 💡 6. Key Insights

* Probability of winning depends heavily on **CRR**, **RRR**, and **wickets in hand**.  
* Logistic Regression provides interpretable probabilities and good baseline accuracy.  
* Feature engineering is crucial — especially for real-time sports prediction.  
* The project can easily extend to **live match analytics** or **dashboard applications**.

---

## 🧰 7. Tools and Technologies

**Language:** Python 3.10+

**Libraries Used:**

* `pandas` — data manipulation  
* `numpy` — numerical computation  
* `matplotlib` — data visualization  
* `scikit-learn` — machine learning and pipelines  

**Environment:** Jupyter Notebook / Kaggle Notebook

---

## 🚀 8. Future Enhancements

* Integrate **Streamlit dashboard** for live match win predictions.  
* Add **ensemble models** (Random Forest, XGBoost) for higher accuracy.  
* Fetch **live IPL data via API** for real-time insights.  
* Visualize momentum swings using animated plots.

---

## 👨‍💻 9. Author

**Name:** Arnav Singh  
**Role:** Machine Learning Enthusiast  
**Objective:** To apply data-driven learning and machine intelligence to real-world predictive systems.

📬 **Contact:**

* Email:  itsarnav.singh80@gmail.com  
* LinkedIn: [https://www.linkedin.com/in/arnav-singh-a87847351](https://www.linkedin.com/in/arnav-singh-a87847351)  
* GitHub: [https://github.com/Arnav-Singh-5080](https://github.com/Arnav-Singh-5080)

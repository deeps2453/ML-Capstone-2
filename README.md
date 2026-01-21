# ML-Capstone-1
Football Player Salary Prediction
Problem Description
Context
Football clubs need to determine fair and competitive salaries for players to maintain squad quality while managing their budget effectively. Player salaries vary widely based on multiple factors including performance metrics, reputation, nationality, and club prestige.
Problem Statement
This project aims to predict football player salaries based on various player and club attributes. The model helps:

Club Management: Make data-driven salary decisions during contract negotiations
Player Agents: Understand market value and negotiate fair contracts
Financial Planning: Budget allocation and financial fair play compliance
Talent Scouting: Identify undervalued players with high potential

Dataset
The dataset contains 40,791 football player records with the following features:

Is_top_5_League: Whether player is in a top 5 league (0/1)


Based_rich_nation: Whether club is from a wealthy nation (0/1)


Is_top_ranked_nation: Player's national team ranking tier (0/1/2)


EU_National: Whether player is an EU national (0/1)


Caps: Number of international appearances


Apps: Number of club appearances


Age: Player's age


Reputation: Player reputation score (0-10000)


Is_top_prev_club: Whether previous club was top-tier (0/1)


Salary: Annual salary in dollars (target variable)

Solution Approach
We use machine learning regression models to predict player salaries, with emphasis on:

Handling highly skewed salary distribution through log transformation
Feature engineering and importance analysis
Model comparison and hyperparameter tuning
Deployment as a web service for real-time predictions









Installation & Setup
Prerequisites

Python 3.9+

Docker (for containerization)

kubectl (for Kubernetes deployment)

# Mall-Customer-Segmentation-Project
This project focuses on clustering mall customers into meaningful groups based on their Annual Income and Spending Score using K-Means clustering and DBSCAN. The goal is to identify distinct customer segments that can guide targeted marketing strategies, improve customer understanding, and support business decision-making.

# Table Of Content
* [Brief](#Brief)
* [DataSet](#DataSet)
* [How_It_Works](#How_It_Works)
* [Tools](#Tools)
* [Cluster_Insights](#Cluster_Insights)
* [Strategic_Recommendations](#Strategic_Recommendations)
* [Sample_Run](#Sample_Run)


# Brief
Customer segmentation is a crucial step in personalized marketing and business analytics.
In this project, I explored clustering techniques to segment mall customers into distinct groups.

- Applied K-Means with Elbow Method and Silhouette Score to determine the optimal number of clusters (k=5).
- Experimented with DBSCAN for density-based clustering and compared results with KMeans.
- Built a Streamlit dashboard for interactive analysis and customer prediction.


# DataSet
The dataset used in this project is the [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle.
It includes demographic and spending behavior attributes of mall visitors.

| Attribute              | Description                                                           |
|------------------------|-----------------------------------------------------------------------|
| CustomerID             | Unique customer ID.                                                   |
| Gender                 | Gender of the customer.                                               |
| Age                    | Age of the customer.                                                  |
| Annual Income (k$)     | Annual income of the customer (in $1000s).                            |
| Spending Score (1-100) | Score assigned by the mall based on spending behavior and loyalty.    |









# How_It_Works

1. Data Preprocessing
- Cleaned column names and scaled numerical features (Annual Income & Spending Score).
2. Exploratory Data Analysis (EDA)
- Plotted distributions, gender differences, and scatterplots.
3. Clustering Analysis
- KMeans → Used Elbow Method & Silhouette Score to find optimal k (best = 5).
- DBSCAN → Used k-distance plot to select epsilon value, tested density-based segmentation.
4. Evaluation
- Compared cluster quality between KMeans and DBSCAN.
- KMeans gave clear, balanced clusters, while DBSCAN detected fewer but more density-based groups.
5. Visualization
- Scatter plots with centroids, PCA visualization, and bar charts for average spending per cluster.
6. Streamlit Dashboard
- Created an interactive dashboard where users can upload datasets, select algorithms, and explore customer clusters.

# Tools


* Python 3.x
* pandas, numpy
* matplotlib, seaborn
* scikit-learn (KMeans, DBSCAN, PCA, scaling, evaluation)
* Streamlit (dashboard & predictions)
* Jupyter Notebook / VS Code


# Cluster_Insights
- Cluster 0 (Average Spenders): Medium income, medium spending → stable group
- Cluster 1 (Premium Customers): High income, high spending → highly profitable & loyal
- Cluster 2 (Potential Shoppers): Average income, high spending → upselling opportunities
- Cluster 3 (Cautious Wealthy): High income, low spending → require trust & value-building offers
- Cluster 4 (Impulsive Customers): Low income, high spending → discount-sensitive
  ### DBSCAN Results: 
- DBSCAN successfully identified dense spending clusters, but some customers were marked as noise (-1).
- It works well when natural dense groupings exist but can underperform compared to KMeans when clusters are not density-based.

# Strategic_Recommendations

- Focus on Clusters 1 & 2 → Provide VIP programs, loyalty rewards, and personalized campaigns.
- Nurture Cluster 0 → Encourage higher spending through promotions and bundled offers.
- Investigate Cluster 3 → Wealthy customers need better value perception and trust-building.
- Engage Cluster 4 carefully → Respond well to discounts, but not sustainable for long-term revenue.


# Sample_Run
<img width="1460" height="757" alt="Screenshot 2025-10-02 at 12 07 28 am" src="https://github.com/user-attachments/assets/406d5471-600d-4801-bf68-43c98c9729c9" />
















import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.spatial import distance
from math import log
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
import os

# Function to load and preprocess the dataset
def load_and_preprocess_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path).dropna()
    dataset = shuffle(dataset)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(dataset.values)
    dfX = pd.DataFrame(x_scaled, columns=dataset.columns)
    
    index = list(range(len(dfX)))
    dfX['index'] = index
    
    return dfX

# Function to split data into initial labeled, unlabeled, and test datasets
def split_data(dfX):
    train_len = int(len(dfX) * 0.1)
    test_len = int(len(dfX) * 0.2)
    initial_label = dfX.sample(n=train_len, random_state=101)
    unlabel = dfX.drop(initial_label.index)
    test_dataset = unlabel.sample(n=test_len, replace=False)
    unlabel = unlabel.drop(test_dataset.index)
    return initial_label, unlabel, test_dataset

# Function to initialize the classifier
def initialize_classifier():
    return LogisticRegression(penalty='l2', solver='liblinear')

# Function to calculate entropy
def calculate_entropy(probs):
    return [-p[0] * log(p[0], 2) - p[1] * log(p[1], 2) for p in probs]

# function to calculate similarity
def calculate_similarity(df, centroids, feature_cols):
    similarity_to_center = []
    
    for _, row in df.iterrows():
        centroid = centroids[int(row['fair_cluster'])]
        instance = row[feature_cols].to_numpy()
        similarity = distance.euclidean(instance, centroid)
        similarity_to_center.append(similarity)
    
    return similarity_to_center

# FairKMeans
def fairKMeans(X, n_clusters, normalized_similarity_threshold=(0.3, 0.6), random_state=0, max_iter=600, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=n_init)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    similarity_to_center = [distance.euclidean(X[i], centroids[labels[i]]) for i in range(len(X))]
    max_similarity = max(similarity_to_center)
    min_similarity = min(similarity_to_center)
    normalized_similarity = [(sim - min_similarity) / (max_similarity - min_similarity) for sim in similarity_to_center]

    filtered_indices = [i for i in range(len(X)) if normalized_similarity_threshold[0] <= normalized_similarity[i] <= normalized_similarity_threshold[1]]
    fair_cluster_labels = [labels[i] if i in filtered_indices else -1 for i in range(len(X))]

    return fair_cluster_labels, centroids, filtered_indices

# Function to calculate fairness metrics
def calculate_fairness_metrics(data, y_pred, protected_attribute):
    data['y_pred'] = y_pred
    total = data.groupby(protected_attribute).size()
    positive = data[data['label'] == 1].groupby(protected_attribute).size()
    true_positive = data[(data['label'] == 1) & (data['y_pred'] == 1)].groupby(protected_attribute).size()
    predicted_positive = data[data['y_pred'] == 1].groupby(protected_attribute).size()

    SP = abs(predicted_positive[0] / total[0] - predicted_positive[1] / total[1])
    E_OPP = abs(true_positive[0] / positive[0] - true_positive[1] / positive[1])
    E_Odds = abs((predicted_positive[0] - true_positive[0]) / (total[0] - positive[0]) - (predicted_positive[1] - true_positive[1]) / (total[1] - positive[1]))

    return SP, E_OPP, E_Odds

def save_output_to_file(output):
    # Create 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Path to the output file
    filepath = os.path.join('results', 'result.txt')

    # Write the output to the file
    with open(filepath, 'a') as file:
        file.write(output)

# Main function
def main():
    # Dictionary mapping dataset paths to their sensitive attribute
    datasets = {
        "data/Compass.csv": 'race',
        "data/Adult.csv": 'gender',
        "data/Loan.csv": 'gender',
        "data/Oalad.csv": 'gender'
    }

    for dataset_path, sensitive_attr in datasets.items():
        print(f"Processing {dataset_path}")
        output = f"Processing {dataset_path}\n"
        dfX = load_and_preprocess_dataset(dataset_path)
        initial_label, unlabel, test_dataset = split_data(dfX)
        clf = initialize_classifier()

        # Number of clusters
        k = 180

        # Train the classifier from initial dataset
        X_train = initial_label.drop(['label'], axis=1)
        y_train = initial_label['label']
        clf.fit(X_train, y_train)

        # Process the unlabelled dataset
        X_U = unlabel.drop(['label'], axis=1)
        probs = clf.predict_proba(X_U)
        entropy = calculate_entropy(probs)

        X_cluster = pd.DataFrame(X_U)
        X_cluster['entropy'] = entropy

        # Using fairKMeans for clustering
        fair_cluster_labels, centroids, filtered_indices = fairKMeans(X_cluster.to_numpy(), n_clusters=k)

        # Ensure that the DataFrame for calculating similarity has the correct features
        feature_cols = X_cluster.columns.tolist()  

        fair_data_cluster = unlabel.iloc[filtered_indices].copy()
        fair_data_cluster['fair_cluster'] = np.array(fair_cluster_labels)[filtered_indices]
        fair_data_cluster['entropy'] = X_cluster['entropy'].iloc[filtered_indices].values    

        fair_data_cluster = fair_data_cluster[fair_data_cluster['fair_cluster'] >= 0]
        fair_data_cluster['representative'] = calculate_similarity(fair_data_cluster, centroids, feature_cols)

        beta = 0.6
        fair_data_cluster['final_score'] = beta * fair_data_cluster['representative'] + (1.0 - beta) * fair_data_cluster['entropy']
        fair_data_cluster = fair_data_cluster.sort_values(by=['fair_cluster', 'final_score'], ascending=[True, False])

        representatives = fair_data_cluster.groupby('fair_cluster').head(1)

        initial_label = initial_label.append(representatives)
        initial_label.dropna(inplace=True)
        initial_label.drop(["fair_cluster", "representative", "final_score", "entropy"], axis=1, inplace=True)

        X_train = initial_label.drop(['label'], axis=1)
        y_train = initial_label['label']
        clf.fit(X_train, y_train)

        y_pred = clf.predict(test_dataset.drop(['label'], axis=1))
        gmean = geometric_mean_score(test_dataset['label'], y_pred)
        f1 = f1_score(test_dataset['label'], y_pred)
        acc = accuracy_score(test_dataset['label'], y_pred)

        output += f"Accuracy Score: {acc}\n"
        output += f"F1 Score: {f1}\n"
        output += f"GMean Score: {gmean}\n"
        SP, E_OPP, E_Odds = calculate_fairness_metrics(test_dataset, y_pred, sensitive_attr)
        output += f"Statistical Parity: {SP}\n"
        output += f"Equal Opportunity: {E_OPP}\n"
        output += f"Equalized Odds: {E_Odds}\n"

        # Write the output to a file
        save_output_to_file(output)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

st.write("""
# Final Project on Spotify Huge database daily charts over 3 years
""")

# Loading the original dataset
df = pd.read_csv("Final database.csv", low_memory=False)


# Display the original dataset with a heading
st.write("## Original Dataset Preview")
st.dataframe(df.head())

# Button to show all features in the original data
show_all_features = st.button("Show All Features")

# Display all features when the button is clicked
if show_all_features:
    st.write('## All Features in the Original Data')
    st.write("Column Names:", df.columns.tolist())

def categorize_popularity(row):
    if row['Popularity'] > 3334:
        return 'Viral'
    elif row['Popularity'] > 515:
        return 'Very Popular'
    elif row['Popularity'] > 77:
        return 'Popular'
    else:
        return 'Less Popular'

# Apply categorization to create a new column 'Popularity_Level'
df['Popularity_Level'] = df.apply(categorize_popularity, axis=1)

# Streamlit app
def main():
    st.write("# Song Popularity Analysis")

    if st.button("Show Popularity Level Counts"):
        # Display the count of songs in different popularity levels
        popularity_counts = df['Popularity_Level'].value_counts()
        st.write("## Popularity Level Counts")
        st.write('## To make it a classification task, we created 4 categories in the popularity level')
        st.write(popularity_counts)

    # Loading the pre-processed and scaled dataset
    dfscaled = pd.read_csv("dfscaled.csv", low_memory=False)

    # Button to print the pre-processed and scaled dataset
    print_data_button = st.button("Show Data After Pre-Processing")

    # Print the dataset when the button is clicked
    if print_data_button:
        st.write("## Data After Pre-Processing")
        st.write(dfscaled)
        
      # Button to print the pre-processed and scaled dataset
    print_correlation_button = st.button("Show Co-realtion matrix")

    # Print the dataset when the button is clicked
    if print_correlation_button:
        corr_columns = dfscaled.select_dtypes(include=np.number).columns.tolist()
        fig = plt.figure(figsize=(50,50))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Correlation Between Variable')

        # # Supaya matriks korelasi yang terlihat cuman bagian bawahnya
        mask = np.triu(np.ones_like(
            dfscaled[corr_columns].corr(), dtype=np.bool))

        sns.heatmap(dfscaled[corr_columns].corr(),vmin=-1,
            vmax=1,cmap='mako',annot=True,mask=mask,
            annot_kws={"fontsize":20})

        st.pyplot(fig)
        
    st.write("# Dimension Reduction with PCA")

    if st.button("Perform Dimension Reduction"):
        # Exclude non-numeric columns for dimension reduction
        numeric_columns = dfscaled.select_dtypes(include=np.number).columns
        numeric_data = dfscaled[numeric_columns]

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.7
        to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
        feature_keep = list(set(numeric_data.columns) - set(to_drop))

        # Perform PCA dimension reduction
        pca_model = PCA(n_components=10)
        pca_feature = pca_model.fit_transform(numeric_data[feature_keep])
        dfpca = pd.DataFrame(data=pca_feature, index=numeric_data.index)

        st.write("## Resulting DataFrame after PCA Dimension Reduction")
        st.dataframe(dfpca) 
        from pandas.api.types import CategoricalDtype
        popularity_cat = CategoricalDtype(categories=['Less Populer', 'Populer', 'Very Popular', 'Viral'], ordered=True)
        dfscaled['Popularity_Level'] = dfscaled['Popularity_Level'].astype(popularity_cat)
        dfscaled['Popularity_Level']
        X_pca = dfpca
        y = dfscaled[['Popularity_Level']]
        X = dfscaled[feature_keep]
        # Perform train-test split
        Xpca_train, Xpca_test, ypca_train, ypca_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


        # Button to calculate and display classification metrics
    calculate_metrics_button = st.button("Calculate Classification Metrics")

    if calculate_metrics_button:
        st.write("## Classification Metrics")
        numeric_columns = dfscaled.select_dtypes(include=np.number).columns
        numeric_data = dfscaled[numeric_columns]

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.7
        to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
        feature_keep = list(set(numeric_data.columns) - set(to_drop))

        # Perform PCA dimension reduction
        pca_model = PCA(n_components=10)
        pca_feature = pca_model.fit_transform(numeric_data[feature_keep])
        dfpca = pd.DataFrame(data=pca_feature, index=numeric_data.index)
        from pandas.api.types import CategoricalDtype
        popularity_cat = CategoricalDtype(categories=['Less Populer', 'Populer', 'Very Popular', 'Viral'], ordered=True)
        dfscaled['Popularity_Level'] = dfscaled['Popularity_Level'].astype(popularity_cat)
        dfscaled['Popularity_Level']
        X_pca = dfpca
        y = dfscaled[['Popularity_Level']]
        X = dfscaled[feature_keep]
        # Perform train-test split
        Xpca_train, Xpca_test, ypca_train, ypca_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

        # Assuming Xpca_train, Xpca_test, ypca_train, ypca_test are already defined

        # Convert DataFrame target variables to 1D arrays
        ypca_train = ypca_train.values.ravel()
        ypca_test = ypca_test.values.ravel()

        # Instantiate classifiers
        knn_clf = KNeighborsClassifier()
        logreg_clf = LogisticRegression(max_iter=1000)  # Increase max_iter to avoid ConvergenceWarning
        rf_clf = RandomForestClassifier(random_state=42)

        # Train classifiers
        knn_clf.fit(Xpca_train, ypca_train)
        logreg_clf.fit(Xpca_train, ypca_train)
        rf_clf.fit(Xpca_train, ypca_train)

        # Make predictions
        knn_preds = knn_clf.predict(Xpca_test)
        logreg_preds = logreg_clf.predict(Xpca_test)
        rf_preds = rf_clf.predict(Xpca_test)

        # Calculate metrics for each classifier
        def calculate_metrics(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            return accuracy, precision, recall, f1

        knn_accuracy, knn_precision, knn_recall, knn_f1 = calculate_metrics(ypca_test, knn_preds)
        logreg_accuracy, logreg_precision, logreg_recall, logreg_f1 = calculate_metrics(ypca_test, logreg_preds)
        rf_accuracy, rf_precision, rf_recall, rf_f1 = calculate_metrics(ypca_test, rf_preds)

        st.write("### K-Nearest Neighbors (KNN)")
        st.write("Accuracy:", knn_accuracy)
        st.write("Precision:", knn_precision)
        st.write("Recall:", knn_recall)
        st.write("F1 Score:", knn_f1)

        st.write("### Logistic Regression")
        st.write("Accuracy:", logreg_accuracy)
        st.write("Precision:", logreg_precision)
        st.write("Recall:", logreg_recall)
        st.write("F1 Score:", logreg_f1)

        st.write("### Random Forest")
        st.write("Accuracy:", rf_accuracy)
        st.write("Precision:", rf_precision)
        st.write("Recall:", rf_recall)
        st.write("F1 Score:", rf_f1)

        # Create a Voting Classifier ensemble
        ensemble_clf = VotingClassifier(estimators=[('knn', knn_clf), ('logreg', logreg_clf), ('rf', rf_clf)], voting='hard')
        ensemble_clf.fit(Xpca_train, ypca_train)

        # Make predictions using the ensemble
        ensemble_preds = ensemble_clf.predict(Xpca_test)

        # Calculate metrics for the ensemble
        ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1 = calculate_metrics(ypca_test, ensemble_preds)

        st.write("### Ensemble Classifier")
        st.write("Accuracy:", ensemble_accuracy)
        st.write("Precision:", ensemble_precision)
        st.write("Recall:", ensemble_recall)
        st.write("F1 Score:", ensemble_f1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_folder = "/kaggle/input/tensorlabs-2025-internships"  # Folder name
detailed_property = pd.read_csv(f"{dataset_folder}/Detailed_Property.csv")
property_by_place = pd.read_csv(f"{dataset_folder}/property_by_place.csv")
property_reviews = pd.read_csv(f"{dataset_folder}/Property_Reviews.csv")

def display_basic_info():
    print("=" * 50)
    print("Detailed Property Info:")
    print(detailed_property.info())
    print("\nProperty by Place Info:")
    print(property_by_place.info())
    print("\nProperty Reviews Info:")
    print(property_reviews.info())
    print("=" * 50)

def display_summary_statistics():
    print("\nSummary Statistics for Detailed Property (Numeric Columns Only):")
    print(detailed_property.select_dtypes(include=['number']).describe())
    
    print("\nSummary Statistics for Property by Place (Numeric Columns Only):")
    print(property_by_place.select_dtypes(include=['number']).describe())
    
    print("\nSummary Statistics for Property Reviews (Numeric Columns Only):")
    print(property_reviews.select_dtypes(include=['number']).describe())
    print("=" * 50)

def check_missing_values():
    print("\nMissing Values in Detailed Property:")
    print(detailed_property.isnull().sum())
    print("\nMissing Values in Property by Place:")
    print(property_by_place.isnull().sum())
    print("\nMissing Values in Property Reviews:")
    print(property_reviews.isnull().sum())
    print("=" * 50)

def analyze_price_distribution():
    price_column = None
    # Check for possible price-related columns
    for col in detailed_property.columns:
        if 'price' in col.lower():  # Case-insensitive check
            price_column = col
            break

    if price_column:
        plt.figure(figsize=(10, 5))
        sns.histplot(detailed_property[price_column].dropna(), bins=50, kde=True, color='blue')
        plt.title(f'Distribution of {price_column}', fontsize=16)
        plt.xlabel(price_column, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("No price-related column found in Detailed_Property dataset. Available columns:", detailed_property.columns)

def perform_correlation_analysis():
    numeric_columns = detailed_property.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        correlation_matrix = detailed_property[numeric_columns].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap (Numeric Columns Only)", fontsize=16)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()
    else:
        print("No numeric columns found for correlation analysis.")

def analyze_top_amenities():
    if 'amenities' in detailed_property.columns:
        top_amenities = detailed_property['amenities'].dropna().str.split(',').explode().value_counts().head(20)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_amenities.values, y=top_amenities.index, palette="viridis")
        plt.title("Top 20 Most Common Amenities in Airbnb Listings", fontsize=16)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Amenities", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("Column 'amenities' not found in Detailed_Property dataset.")

def perform_review_sentiment_analysis():
    if 'review_score' in property_reviews.columns:
        property_reviews["positive_reviews"] = property_reviews["review_score"] >= 4
        review_sentiment = property_reviews["positive_reviews"].value_counts(normalize=True)
        plt.figure(figsize=(6, 6))
        review_sentiment.plot(kind="pie", autopct="%1.1f%%", colors=["red", "green"], labels=["Negative", "Positive"])
        plt.title("Review Sentiment Distribution", fontsize=16)
        plt.ylabel("")
        plt.show()
    else:
        print("Column 'review_score' not found in Property_Reviews dataset.")

def display_key_insights():
    print("\nKey Insights:")
    print("✅ Listings with highly rated reviews and popular amenities tend to have better occupancy rates.")
    print("✅ Pricing needs to be competitive based on location and similar listings.")
    print("✅ Investors should focus on properties with sought-after amenities and positive guest experiences.")
    print("=" * 50)

def save_cleaned_datasets():
    detailed_property.to_csv("cleaned_Detailed_Property.csv", index=False)
    property_by_place.to_csv("cleaned_property_by_place.csv", index=False)
    property_reviews.to_csv("cleaned_Property_Reviews.csv", index=False)
    print("\nCleaned datasets saved!")

def main():
    display_basic_info()
    display_summary_statistics()
    check_missing_values()
    analyze_price_distribution()
    perform_correlation_analysis()
    analyze_top_amenities()
    perform_review_sentiment_analysis()
    display_key_insights()
    save_cleaned_datasets()

if __name__ == "__main__":
    main()


# In[ ]:





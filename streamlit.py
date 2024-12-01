import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Load the data
best_seller = pd.read_excel('https://raw.githubusercontent.com/RifaldiAchmad/Data-Analysis-with-Project/data/refs/heads/main/Dataset%20Clean/best_seller.xlsx')

# Preprocessing data
best_seller['order_purchase_timestamp'] = pd.to_datetime(best_seller['order_purchase_timestamp'])

# Define the dashboard
st.title("Best Seller Product Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Top Categories", "Sales Trends", "Customer Locations", "RFM Analysis"])

if page == "Top Categories":
    st.header("Top Product Categories")

    # Best-selling products by category
    categories = best_seller['product_category_name_english'].value_counts()
    colors = ["#72BCD4"] + ["#D3D3D3"] * (len(categories) - 1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=categories.index, y=categories.values, palette=colors, ax=ax)
    ax.set_ylabel('Count')
    ax.set_title('Top Product Categories (Best Seller)', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, which='minor', linestyle='--', alpha=0.2)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    st.pyplot(fig)

elif page == "Sales Trends":
    st.header("Sales and Revenue Trends for Best-Selling Products")

    # Filter data: last year only
    last_year = best_seller['order_purchase_timestamp'].max() - pd.DateOffset(years=1)
    filtered_data = best_seller[best_seller['order_purchase_timestamp'] > last_year]

    # Get top category
    top_category = filtered_data['product_category_name_english'].value_counts().idxmax()
    filtered_top_category = filtered_data[filtered_data['product_category_name_english'] == top_category]

    # Group by month
    monthly_data = filtered_top_category.groupby(
        filtered_top_category['order_purchase_timestamp'].dt.to_period('M')).agg(
        total_order=('order_id', 'count'),
        total_sales=('price', 'sum'))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i, (col, ylabel) in enumerate(zip(['total_order', 'total_sales'], ['Total Orders', 'Total Sales ($)'])):
        sns.lineplot(data=monthly_data, x=monthly_data.index.to_timestamp(), y=col, marker='o', ax=axes[i])
        axes[i].set_ylabel(ylabel)
        axes[i].set_title('Total Orders and Total Sales of Best-Selling Products in the Last Year' if i == 0 else '',
                          fontsize=16)
        axes[i].grid(True, linewidth=0.3, alpha=0.5)
        axes[i].spines[['top', 'right']].set_visible(False)
        axes[i].set_xlabel('')
        for x, y in zip(monthly_data.index.to_timestamp(), monthly_data[col]):
            axes[i].text(x, y, f'{y:,.0f}', fontsize=8, ha='center', va='bottom', color='black')

    st.pyplot(fig)

elif page == "Customer Locations":
    st.header("Top 10 Customer Locations")

    # City-wise order counts
    city_order_counts = (
        best_seller.groupby('customer_city')['order_id']
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='customer_city', y='order_id', data=city_order_counts, palette='viridis', ax=ax)
    ax.set_ylabel('Order Count')
    ax.set_title('Top 10 Customer Cities by Order Count (Best Seller)', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.2)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    st.pyplot(fig)

elif page == "RFM Analysis":
    st.header("Customer Segmentation Based on RFM Analysis")

    # RFM analysis
    rfm_df = best_seller.groupby("customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"})
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

    # Calculate recency
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = best_seller["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    # Normalize and calculate RFM score
    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)
    rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100
    rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']
    rfm_df['RFM_score'] *= 0.05

    # Segment customers
    rfm_df["customer_segment"] = np.where(
        rfm_df['RFM_score'] > 4.5, "Top customers", (np.where(
            rfm_df['RFM_score'] > 4, "High value customer", (np.where(
                rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
                    rfm_df['RFM_score'] > 1.6, 'Low value customers', 'Lost customers'))))))
    customer_segment_df = rfm_df.groupby("customer_segment", as_index=False)["customer_id"].nunique()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="customer_segment", y="customer_id", data=customer_segment_df, palette="coolwarm", ax=ax)
    ax.set_title("Number of Customers by Segment", fontsize=16)
    ax.set_ylabel('Number of Customers')
    ax.set_xlabel('Customer Segment')
    st.pyplot(fig)
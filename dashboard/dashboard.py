import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

sns.set(style="darkgrid")

# Load dataset
best_seller = pd.read_excel('https://raw.githubusercontent.com/RifaldiAchmad/Data-Analysis-and-Visualization/refs/heads/main/data/best_seller.csv')

# Convert datetime column to datetime type
best_seller['order_purchase_timestamp'] = pd.to_datetime(best_seller['order_purchase_timestamp'])

def create_monthly_data(best_seller):
    last_year = best_seller['order_purchase_timestamp'].max() - pd.DateOffset(years=1)
    filtered_data = best_seller[best_seller['order_purchase_timestamp'] > last_year]
    top_category = filtered_data['product_category_name_english'].value_counts().idxmax()
    filtered_top_category = filtered_data[filtered_data['product_category_name_english'] == top_category]
    monthly_data = filtered_top_category.groupby(filtered_top_category['order_purchase_timestamp'].dt.to_period('M')).agg(
        total_sales=('price', 'sum'))
    
    # Convert PeriodIndex to string for better visualization in Streamlit
    monthly_data.index = monthly_data.index.astype(str)
    
    return monthly_data

def create_city_order_counts(best_seller):
    city_order_counts = (
        best_seller.groupby('customer_city')['order_id']
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index())
    return city_order_counts

def create_rfm_df(best_seller):
    rfm_df = best_seller.groupby("customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", 
        "order_id": "nunique", 
        "price": "sum"})
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = best_seller["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    return rfm_df

def create_customer_segment_df(rfm_df):
    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)
    rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*100
    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)
    rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']
    rfm_df['RFM_score'] *= 0.05
    rfm_df = rfm_df.round(2)
    rfm_df["customer_segment"] = np.where(
        rfm_df['RFM_score'] > 4.5, "Top customers", np.where(
        rfm_df['RFM_score'] > 4, "High value customer", np.where(
        rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
        rfm_df['RFM_score'] > 1.6, 'Low value customers', 'lost customers'))))
    customer_segment_df = rfm_df.groupby("customer_segment", as_index=False)["customer_id"].nunique().sort_values("customer_id", ascending=False).reset_index(drop=True)
    customer_segment_df['customer_segment'] = pd.Categorical(customer_segment_df['customer_segment'], [
        "lost customers", "Low value customers", "Medium value customer",
        "High value customer", "Top customers"])
    return customer_segment_df

monthly_data = create_monthly_data(best_seller)
city_order_counts = create_city_order_counts(best_seller)
rfm_df = create_rfm_df(best_seller)
customer_segment_df = create_customer_segment_df(rfm_df)

# Streamlit App
st.title("Best Seller Dashboard")

# Sidebar for date and category selection
st.sidebar.title("Filter Data")
min_date = best_seller['order_purchase_timestamp'].min().date()
max_date = best_seller['order_purchase_timestamp'].max().date()

start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
selected_categories = st.sidebar.multiselect(
    'Select Categories',
    options=best_seller['product_category_name_english'].unique(),
    default=best_seller['product_category_name_english'].unique()
)

# Filter data based on selected categories and date range
filtered_best_seller = best_seller[
    (best_seller['order_purchase_timestamp'].dt.date >= start_date) & 
    (best_seller['order_purchase_timestamp'].dt.date <= end_date) &
    (best_seller['product_category_name_english'].isin(selected_categories))
]

# Create new visualizations based on filtered data
monthly_data = create_monthly_data(filtered_best_seller)
city_order_counts = create_city_order_counts(filtered_best_seller)
rfm_df = create_rfm_df(filtered_best_seller)
customer_segment_df = create_customer_segment_df(rfm_df)

st.header("Monthly Performance of Top Category")
st.line_chart(monthly_data)

st.header("Top 10 Cities by Orders")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='order_id', y='customer_city', data=city_order_counts, palette='coolwarm', ax=ax)
ax.set_xlabel(" ")
ax.set_ylabel(" ")
st.pyplot(fig)

st.header("Customer Segmentation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="customer_id", y="customer_segment", data=customer_segment_df, palette="viridis", ax=ax)
ax.set_xlabel(" ")
ax.set_ylabel(" ")
st.pyplot(fig)

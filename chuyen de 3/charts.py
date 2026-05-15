import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(
    style="whitegrid",
    context="paper"
)

# =====================================================
# 1. DOANH THU THEO THÁNG
# =====================================================
def chart_monthly_sales(df):

    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(7,4)
    )

    sns.lineplot(
        data=monthly_sales,
        x="Month",
        y="Sales",
        marker="o",
        linewidth=3,
        ax=ax
    )

    ax.set_title(
        "Doanh thu theo tháng",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 2. TOP PRODUCTS
# =====================================================
def chart_top_products(df):

    top_products = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(7,4)
    )

    sns.barplot(
        data=top_products,
        x="Sales",
        y="Product Name",
        palette="Blues_r",
        ax=ax
    )

    # VALUE LABEL
    for p in ax.patches:

        width = p.get_width()

        ax.annotate(
            f"{width:,.0f}",

            (
                width,
                p.get_y() + p.get_height()/2
            ),

            ha="left",
            va="center",
            fontsize=8
        )

    ax.set_title(
        "Top 10 sản phẩm",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 3. REGION SALES
# =====================================================
def chart_region_sales(df):

    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(6,6)
    )

    ax.pie(
        region_sales.values,
        labels=region_sales.index,
        autopct="%1.1f%%"
    )

    ax.set_title(
        "Doanh thu theo khu vực",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 4. PROFIT BY CATEGORY
# =====================================================
def chart_profit_category(df):

    profit_df = (
        df.groupby(
            ["Category", "Sub-Category"]
        )["Profit"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.barplot(
        data=profit_df,
        x="Profit",
        y="Sub-Category",
        hue="Category",
        dodge=False,
        ax=ax
    )

    ax.axvline(
        0,
        color="black"
    )

    ax.set_title(
        "Lợi nhuận theo nhóm hàng",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 5. PROFIT MARGIN
# =====================================================
def chart_profit_margin(df):

    margin_df = (
        df.groupby("Sub-Category")[["Sales", "Profit"]]
        .sum()
    )

    margin_df["Margin"] = (
        margin_df["Profit"] /
        margin_df["Sales"]
    ) * 100

    margin_df = (
        margin_df.sort_values(
            "Margin",
            ascending=False
        )
        .reset_index()
    )

    colors = [
        "#2ecc71"
        if x > 0
        else "#e74c3c"
        for x in margin_df["Margin"]
    ]

    fig, ax = plt.subplots(
        figsize=(9,5)
    )

    sns.barplot(
        data=margin_df,
        x="Margin",
        y="Sub-Category",
        palette=colors,
        ax=ax
    )

    ax.axvline(
        0,
        color="black"
    )

    ax.set_title(
        "Tỷ suất lợi nhuận (%)",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 6. DISCOUNT VS PROFIT
# =====================================================
def chart_discount_profit(df):

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    sns.lineplot(
        data=df,
        x="Discount",
        y="Profit",
        marker="o",
        linewidth=3,
        color="red",
        errorbar=None,
        ax=ax
    )

    ax.axhline(
        0,
        linestyle="--",
        color="black"
    )

    ax.set_title(
        "Giảm giá vs Lợi nhuận",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 7. CUSTOMER SEGMENT
# =====================================================
def chart_segment(df):

    segment_df = (
        df.groupby("Segment")[["Sales", "Profit"]]
        .sum()
        .reset_index()
    )

    melted = segment_df.melt(
        id_vars="Segment",
        var_name="Metric",
        value_name="Value"
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    sns.barplot(
        data=melted,
        x="Segment",
        y="Value",
        hue="Metric",
        ax=ax
    )

    ax.set_title(
        "Phân khúc khách hàng",
        fontsize=14,
        fontweight="bold"
    )

    return fig


# =====================================================
# 8. TOP VIP CUSTOMERS
# =====================================================
def chart_top_customers(df):

    top_customers = (
        df.groupby("Customer Name")[["Sales", "Profit"]]
        .sum()
        .sort_values(
            "Sales",
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    melted = top_customers.melt(
        id_vars="Customer Name",
        var_name="Metric",
        value_name="Value"
    )

    fig, ax = plt.subplots(
        figsize=(9,5)
    )

    sns.barplot(
        data=melted,
        x="Value",
        y="Customer Name",
        hue="Metric",
        ax=ax
    )

    ax.set_title(
        "Top khách hàng VIP",
        fontsize=14,
        fontweight="bold"
    )

    return fig

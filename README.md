# E-Commerce Sales & Customer Analysis Dashboard (Power BI)

Interactive Power BI dashboard analyzing e-commerce sales, profit, customers, products, and orders. Data is cleaned in **Power Query** and KPIs are built with **DAX**.

## Business questions answered
- How are sales and profit trending month by month and year over year?
- Which categories and products make the most profit, and which have weak margins?
- Who are the top customers, and how many come back to buy again?
- Which payment methods do customers prefer?
- How high are return and cancellation rates?

## Dataset
Synthetic Indian e-commerce data (2023-2025), created for practice:
| File | Rows | Description |
|---|---|---|
| orders.csv | ~4,800 | Order lines: dates, quantity, discount, payment method, status |
| customers.csv | 500 | Name, city, state, segment |
| products.csv | 48 | Category, sub-category, cost, price |

Orders data has deliberate issues (duplicates, text dates, blank discounts, inconsistent payment names) to practise cleaning.

## Data cleaning (Power Query)
Removed duplicates, converted text dates, filled blank discounts, standardised payment method text, set correct data types. See `01_PowerQuery_Steps.md`.

## Data model
Star schema: `Date`, `customers`, `products` (dimensions) linked to `orders` (fact).

## Key DAX measures
Total Sales, Total Profit, Profit Margin %, Avg Order Value, YoY Growth %, Sales YTD, Return Rate %, Repeat Customer %. Full code in `02_DAX_Measures.md`.

## Dashboard pages
1. Sales Overview
2. Product Performance
3. Customer Insights
4. Orders & Payments

*(add screenshots here)*

## Key insights
*(fill in after you build it, e.g. "Electronics gives the highest profit, Books have low sales but the best margin")*

## Tools
Power BI Desktop, Power Query, DAX

## How to run
1. Clone the repo
2. Open the `.pbix` in Power BI Desktop
3. If the data path breaks: Transform Data > Data source settings > change source to the `data` folder

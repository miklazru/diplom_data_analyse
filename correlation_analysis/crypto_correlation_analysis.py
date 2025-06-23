#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import shapiro  # Добавлен новый импорт

# 1. Загрузка данных Ethereum
def load_eth_data():
    """Возвращает DataFrame с метриками Ethereum"""
    data = {
        "Date": pd.to_datetime(['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']),
        "TVL (млрд $)": [40, 60, 80, 90, 100],
        "Transactions (млн)": [0.6, 1.0, 1.5, 1.7, 1.8],
        "Stablecoins (млрд $)": [30, 60, 90, 110, 120],
        "ETH Price ($)": [1, 2, 3, 3.5, 4]
    }
    return pd.DataFrame(data)

# 2. Загрузка данных Bitcoin с правильным временным периодом
def load_btc_data():
    """Загружает данные BTC за те же даты, что и ETH"""
    try:
        dates = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
        btc_prices = []
        for date in dates:
            data = yf.download('BTC-USD', start=date, end=pd.to_datetime(date)+pd.Timedelta(days=1), progress=False)
            if not data.empty:
                btc_prices.append(data['Close'].iloc[0])
            else:
                btc_prices.append(None)
        return btc_prices
    except Exception as e:
        print(f"Ошибка загрузки BTC: {e}")
        return None

# 3. Основной блок выполнения
if __name__ == "__main__":
    # Загрузка данных
    eth_df = load_eth_data()
    btc_prices = load_btc_data()
    
    if btc_prices:
        eth_df['BTC Price ($)'] = btc_prices
    
    # Проверка данных
    print("\nЗагруженные данные:")
    print(eth_df)

    # Проверка нормальности распределения
    print("\nПроверка нормальности распределения:")
    for col in ['TVL (млрд $)', 'Stablecoins (млрд $)']:
        stat, p = shapiro(eth_df[col])
        print(f'{col}: p-value = {p:.3f} (нормальность: {"Да" if p > 0.05 else "Нет"})')

    # Построение scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=eth_df['Stablecoins (млрд $)'], 
        y=eth_df['TVL (млрд $)'],
        hue=eth_df['Date'].dt.year,
        palette='viridis',
        s=100
    )
    plt.title('Связь между TVL и капитализацией стейблкоинов (USDT)', fontsize=14)
    plt.xlabel('Капитализация стейблкоинов (млрд $)', fontsize=12)
    plt.ylabel('TVL (млрд $)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Год')
    plt.tight_layout()
    plt.savefig('tvl_vs_usdt_scatter.png', dpi=200)
    plt.close()
    print("\nScatter plot сохранен как 'tvl_vs_usdt_scatter.png'")

    # Анализ корреляций
    if not eth_df.empty:
        # Расчет корреляции Пирсона для TVL и Stablecoins
        corr_tvl_usdt = eth_df['TVL (млрд $)'].corr(eth_df['Stablecoins (млрд $)'])
        print(f"\nКорреляция TVL-USDT: {corr_tvl_usdt:.3f}")

        # Полная матрица корреляций
        corr_matrix = eth_df.select_dtypes(include='number').corr()
        print("\nМатрица корреляций:")
        print(corr_matrix)
        
        # Визуализация корреляций
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            fmt=".3f",
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Корреляция метрик Ethereum и BTC", pad=20, fontsize=14)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300)
        plt.close()
        print("\nГрафик корреляций сохранен как 'correlation_heatmap.png'")
        
        # Визуализация трендов
        plt.figure(figsize=(14, 8))
        styles = {
            'TVL (млрд $)': {'color': '#1f77b4', 'marker': 'o'},
            'Stablecoins (млрд $)': {'color': '#2ca02c', 'marker': 's'},
            'ETH Price ($)': {'color': '#d62728', 'marker': 'D'},
            'BTC Price ($)': {'color': '#9467bd', 'marker': 'p'}
        }
        
        for column in eth_df.columns[1:]:
            if column != 'Date' and column != 'Transactions (млн)':
                plt.plot(
                    eth_df['Date'], 
                    eth_df[column], 
                    label=column,
                    **styles.get(column, {}),
                    linewidth=2,
                    markersize=8
                )
        
        plt.title('Динамика ключевых метрик (2019-2023)', fontsize=16)
        plt.xlabel('Год', fontsize=14)
        plt.ylabel('Значение', fontsize=14)
        plt.legend(fontsize=12, loc='upper left', ncol=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig('metrics_trends.png', dpi=300)
        plt.close()
        print("График динамики сохранен как 'metrics_trends.png'")
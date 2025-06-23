#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Установка необходимых модулей (раскомментировать при первом запуске)
# import subprocess
# subprocess.run(['pip3', 'install', 'yfinance', 'pandas', 'matplotlib', 'seaborn'])

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. Загрузка и подготовка данных Ethereum
def load_eth_data():
    """Возвращает DataFrame с основными метриками Ethereum"""
    data = {
        "Date": ['2019-01-01', '2021-01-01', '2023-01-01'],
        "TVL (млрд $)": [40, 80, 100],
        "Transactions (млн)": [0.6, 1.5, 1.8],
        "Stablecoins (млрд $)": [30, 90, 120],
        "ETH Price ($)": [1, 3, 4]
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# 2. Загрузка данных Bitcoin
def load_btc_data(start_date):
    """Загружает исторические данные BTC"""
    try:
        btc = yf.download('BTC-USD', start=start_date, progress=False)['Close']
        print("\nДанные BTC успешно загружены!")
        return btc
    except Exception as e:
        print(f"\nОшибка загрузки BTC: {e}")
        return None

# 3. Анализ корреляций
def analyze_correlations(df):
    """Строит матрицу корреляций и тепловую карту"""
    # Расчет корреляций
    corr_matrix = df.corr()
    print("\nМатрица корреляций:")
    print(corr_matrix)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".3f")
    plt.title("Корреляция метрик Ethereum и BTC", pad=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("\nГрафик сохранен как 'correlation_heatmap.png'")

# 4. Основной блок выполнения
if __name__ == "__main__":
    # Загрузка данных
    eth_df = load_eth_data()
    btc_data = load_btc_data(start_date='2019-01-01')
    
    # Объединение данных
    if btc_data is not None:
        # Добавляем цену BTC на соответствующие даты
        eth_df['BTC Price ($)'] = eth_df['Date'].map(
            lambda x: btc_data.loc[btc_data.index.date == x.date()].values[0] if not btc_data.loc[btc_data.index.date == x.date()].empty else None
        )
    
    # Анализ
    analyze_correlations(eth_df.select_dtypes(include='number'))
    
    # Дополнительная визуализация
    plt.figure(figsize=(12, 6))
    for column in eth_df.columns[1:]:
        if column != 'Date':
            plt.plot(eth_df['Date'], eth_df[column], label=column, marker='o')
    plt.title("Динамика метрик Ethereum и BTC")
    plt.xlabel("Дата")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.savefig('metrics_trends.png')
    print("График динамики сохранен как 'metrics_trends.png'")

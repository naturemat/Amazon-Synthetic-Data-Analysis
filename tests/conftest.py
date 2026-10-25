"""Fixtures compartidas para las pruebas del pipeline KDD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_orders() -> pd.DataFrame:
    """Dataset pequeño con los problemas típicos que debe tratar la limpieza."""
    dates = pd.to_datetime([
        "2020-01-01", "2020-01-05", "2020-01-10", "2020-01-10",
        "2020-02-01", "2020-02-15", "2020-03-01", "2020-03-10",
    ])
    customers = ["C1", "C1", "C1", "C2", "C2", "C3", "C4", "C5"]
    products = ["P1", "P2", "P1", "P3", "P1", "P2", "P3", "P1"]
    rows = []
    for i, (date, customer, product) in enumerate(zip(dates, customers, products)):
        quantity = 2 + (i % 3)
        price = 100.0 + i * 10
        discount = [0.0, 0.1, 0.05, 0.2, 0.0, 0.1, 0.0, 0.3][i]
        tax = round(quantity * price * 0.05, 2)
        shipping = 5.0
        total = round(quantity * price + tax + shipping, 2)  # ignora descuento
        rows.append({
            "OrderID": f"O{i + 1}",
            "OrderDate": date,
            "CustomerID": customer,
            "CustomerName": f"Cliente {customer}",
            "ProductID": product,
            "ProductName": f"Producto {product}",
            "Category": "Electronics" if i % 2 == 0 else "Books",
            "Brand": "MarcaA" if i % 2 == 0 else "MarcaB",
            "Quantity": quantity,
            "UnitPrice": price,
            "Discount": discount,
            "Tax": tax,
            "ShippingCost": shipping,
            "TotalAmount": total,
            "PaymentMethod": "Credit Card",
            "OrderStatus": "Delivered" if i < 7 else "Cancelled",
            "City": "Austin",
            "State": "TX",
            "Country": "United States",
            "SellerID": f"S{i % 2 + 1}",
        })
    df = pd.DataFrame(rows)
    # Problemas controlados: un duplicado, un nulo y una cantidad inválida.
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[1, "Discount"] = np.nan
    df.loc[2, "Quantity"] = 0
    return df


@pytest.fixture
def cleaned_orders(raw_orders):
    """Dataset limpio y subconjunto de ventas realizadas."""
    from src.data.data_cleaner import DataCleaner

    cleaner = DataCleaner(raw_orders)
    cleaner.clean()
    return cleaner


@pytest.fixture
def blobs():
    """Dos grupos separados para probar métricas de clustering."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0, scale=0.4, size=(80, 2))
    b = rng.normal(loc=5, scale=0.4, size=(80, 2))
    return np.vstack([a, b])

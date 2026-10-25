"""Pruebas de carga, limpieza y transformación (fases 2-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataLoader
from src.data.transformation import CustomerTransformer


def test_cleaner_handles_duplicates_nulls_and_invalid_rows(raw_orders):
    cleaner = DataCleaner(raw_orders)
    cleaner.clean()

    assert cleaner.report["duplicados_eliminados"] >= 1
    assert cleaner.report["total_nulos_antes"] >= 1
    assert cleaner.report["total_nulos_despues"] == 0
    assert cleaner.report["consistencia"]["filas_eliminadas_cantidad_precio"] >= 1
    assert len(cleaner.data) < len(raw_orders)
    assert cleaner.report["filas_analiticas"] == int(
        cleaner.data["OrderStatus"].isin(["Delivered", "Shipped"]).sum()
    )


def test_cleaner_creates_net_revenue_and_measures_inconsistency(cleaned_orders):
    report = cleaned_orders.report
    revenue = report["ingresos"]
    assert revenue["diferencia_total_amount_vs_neto"] > 0
    assert revenue["filas_inconsistentes_total_amount"] > 0
    assert "Ingreso_Neto" in cleaned_orders.data.columns
    assert "Ingreso_Facturado" in cleaned_orders.data.columns


def test_recency_uses_dataset_cutoff_not_system_date(cleaned_orders):
    sales = cleaned_orders.analytical_data
    cutoff = pd.Timestamp(sales["OrderDate"].max())
    transformer = CustomerTransformer(sales, cutoff_date=cutoff)
    features = transformer.aggregate()

    assert transformer.cutoff_date == cutoff
    assert features["Dias_Ultima_Compra"].min() == 0
    # Si se usara pd.Timestamp.now(), la recencia mínima sería > 0.
    assert features["Dias_Ultima_Compra"].max() <= (
        cutoff - sales["OrderDate"].min()
    ).days


def test_transformation_has_no_nulls_and_scales_features(cleaned_orders):
    sales = cleaned_orders.analytical_data
    transformer = CustomerTransformer(sales)
    features = transformer.aggregate()
    matrix = transformer.prepare_model_features(winsorize=True)

    assert not matrix.isna().any().any()
    assert matrix.shape == (len(features), 5)
    assert np.allclose(matrix.mean().values, 0.0, atol=1e-8)
    assert np.allclose(matrix.std(ddof=0).values, 1.0, atol=1e-8)
    assert transformer.model_values.shape == matrix.shape


def test_loader_profile_detects_structural_signals(tmp_path, raw_orders):
    path = tmp_path / "tiny.csv"
    raw_orders.to_csv(path, index=False)
    loader = DataLoader(str(path))
    df = loader.load()
    profile = loader.profile()

    assert profile["filas"] == len(df)
    assert profile["maximo_items_por_pedido"] >= 1
    assert isinstance(profile["es_probablemente_simulado"], bool)
    saved = loader.save_profile(str(tmp_path / "profile.json"))
    assert (tmp_path / "profile.json").exists()
    assert saved.endswith("profile.json")

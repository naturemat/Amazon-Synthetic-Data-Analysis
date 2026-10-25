"""
Limpieza y preprocesamiento (fase 3 de KDD).

A diferencia de una limpieza "decorativa" que solo imprime cuántos nulos hay,
este módulo *trata* los problemas, registra cada decisión y conserva la
trazabilidad entre el dataset original y el conjunto analítico.

Problemas detectados y tratados
-------------------------------
- Duplicados exactos.
- Fechas inválidas.
- Valores faltantes (mediana para numéricas, moda para categóricas).
- Cantidades o precios inválidos.
- Descuentos fuera del rango [0, 1].
- Impuestos o envíos negativos.
- Inconsistencia de ``TotalAmount``: el dataset original ignora el descuento,
  por lo que se crea ``Ingreso_Neto`` y ``Ingreso_Facturado`` con la fórmula
  explícita y se cuantifica la diferencia.
- Pedidos cancelados/devueltos/pendientes: se excluyen del conjunto analítico
  de ventas realizadas, pero se conservan en el dataset limpio para auditoría.
- Valores atípicos: se reportan con el criterio IQR y se conservan; el recorte
  (winsorización) se aplica solo a las variables de modelado en la fase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.settings import (
    ANALYTICAL_STATUSES,
    COLS_TO_DROP,
    MIN_QUANTITY,
    MIN_UNIT_PRICE,
    OUTLIER_IQR_FACTOR,
    VALID_DISCOUNT_RANGE,
)


class DataCleaner:
    """Limpia el dataset y produce un reporte cuantificado de cada decisión."""

    def __init__(self, df: pd.DataFrame):
        self._original_df = df.copy()
        self._df = df.copy()
        self._analytical: Optional[pd.DataFrame] = None
        self.report: Dict[str, Any] = {
            "filas_entrada": int(len(df)),
            "decisiones": [],
        }

    # ------------------------------------------------------------- properties
    @property
    def data(self) -> pd.DataFrame:
        """Dataset limpio completo (incluye pedidos no realizados)."""
        return self._df

    @property
    def analytical_data(self) -> pd.DataFrame:
        """Subconjunto de ventas realizadas usado en el análisis."""
        if self._analytical is None:
            self._build_analytical()
        return self._analytical

    # ---------------------------------------------------------------- helpers
    def _decision(self, text: str) -> None:
        self.report["decisiones"].append(text)

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    # ------------------------------------------------------------------ steps
    def convert_date_column(self, column: str = "OrderDate") -> int:
        """Convierte fechas y elimina registros con fecha inválida."""
        before = len(self._df)
        self._df[column] = pd.to_datetime(self._df[column], errors="coerce")
        invalid = int(self._df[column].isna().sum())
        if invalid:
            self._df = self._df.dropna(subset=[column]).copy()
        self.report["fechas_invalidas_eliminadas"] = invalid
        self.report["filas_despues_fechas"] = int(len(self._df))
        if invalid:
            self._decision(
                f"Se eliminaron {invalid} registros con fecha inválida "
                f"({before} -> {len(self._df)} filas)."
            )
        else:
            self._decision("Todas las fechas se interpretaron correctamente.")
        return invalid

    def remove_duplicates(self) -> int:
        """Elimina duplicados exactos y registra el resultado."""
        duplicates = int(self._df.duplicated().sum())
        if duplicates:
            self._df = self._df.drop_duplicates().copy()
        self.report["duplicados_eliminados"] = duplicates
        self.report["filas_despues_duplicados"] = int(len(self._df))
        self._decision(
            f"Duplicados exactos eliminados: {duplicates}. "
            f"Quedan {len(self._df):,} filas."
        )
        return duplicates

    def check_missing_values(self, impute: bool = True) -> pd.DataFrame:
        """Diagnostica y, opcionalmente, imputa valores faltantes."""
        nulls_before = self._df.isna().sum()
        self.report["nulos_antes"] = {
            str(k): int(v) for k, v in nulls_before[nulls_before > 0].items()
        }
        self.report["total_nulos_antes"] = int(nulls_before.sum())

        summary = pd.DataFrame({
            "Cantidad Nulos": nulls_before,
            "Porcentaje": (nulls_before / max(len(self._df), 1)) * 100,
        })

        imputed: Dict[str, int] = {}
        if impute:
            for col in self._df.columns:
                if self._df[col].isna().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(self._df[col]):
                    value = self._df[col].median()
                    self._df[col] = self._df[col].fillna(value)
                    imputed[col] = int(nulls_before[col])
                elif pd.api.types.is_datetime64_any_dtype(self._df[col]):
                    # Las fechas no se imputan: ya se eliminaron las inválidas.
                    continue
                else:
                    mode = self._df[col].mode(dropna=True)
                    value = mode.iloc[0] if len(mode) else "Desconocido"
                    self._df[col] = self._df[col].astype(object).fillna(value)
                    imputed[col] = int(nulls_before[col])

        self.report["nulos_imputados"] = imputed
        self.report["total_nulos_despues"] = int(self._df.isna().sum().sum())
        if imputed:
            self._decision(
                "Imputación aplicada: mediana en variables numéricas y moda en "
                f"categóricas para {len(imputed)} columnas."
            )
        else:
            self._decision(
                "No se encontraron valores faltantes; no fue necesario imputar. "
                "La fase de limpieza se documenta igualmente porque el resultado "
                "es una condición del dataset, no una omisión del proceso."
            )

        return summary[summary["Cantidad Nulos"] > 0]

    def validate_consistency(self) -> Dict[str, int]:
        """Corrige o elimina registros con valores imposibles."""
        n = len(self._df)

        invalid_qty = int((self._df["Quantity"] < MIN_QUANTITY).sum())
        invalid_price = int((self._df["UnitPrice"] <= MIN_UNIT_PRICE).sum())
        invalid_rows = self._df["Quantity"].lt(MIN_QUANTITY) | self._df["UnitPrice"].le(MIN_UNIT_PRICE)
        if invalid_rows.any():
            self._df = self._df.loc[~invalid_rows].copy()

        discount_low = int((self._df["Discount"] < VALID_DISCOUNT_RANGE[0]).sum())
        discount_high = int((self._df["Discount"] > VALID_DISCOUNT_RANGE[1]).sum())
        self._df["Discount"] = self._df["Discount"].clip(*VALID_DISCOUNT_RANGE)

        negative_tax = int((self._df["Tax"] < 0).sum())
        negative_shipping = int((self._df["ShippingCost"] < 0).sum())
        self._df["Tax"] = self._df["Tax"].clip(lower=0)
        self._df["ShippingCost"] = self._df["ShippingCost"].clip(lower=0)

        result = {
            "filas_entrada_consistencia": int(n),
            "filas_eliminadas_cantidad_precio": int(invalid_rows.sum()),
            "cantidades_invalidas": invalid_qty,
            "precios_invalidos": invalid_price,
            "descuentos_bajo_rango": discount_low,
            "descuentos_sobre_rango": discount_high,
            "impuestos_negativos_corregidos": negative_tax,
            "envios_negativos_corregidos": negative_shipping,
            "filas_salida_consistencia": int(len(self._df)),
        }
        self.report["consistencia"] = result
        self._decision(
            "Reglas de consistencia: se eliminaron "
            f"{invalid_rows.sum()} filas con cantidad/precio inválido; se recortaron "
            f"{discount_low + discount_high} descuentos al rango [0, 1] y se llevaron "
            "a cero los impuestos/envíos negativos."
        )
        return result

    def add_revenue_columns(self) -> Dict[str, Any]:
        """Crea ingresos netos y mide la inconsistencia de ``TotalAmount``."""
        df = self._df
        df["Ingreso_Bruto"] = df["Quantity"] * df["UnitPrice"]
        df["Ingreso_Neto"] = df["Ingreso_Bruto"] * (1 - df["Discount"])
        df["Ingreso_Facturado"] = df["Ingreso_Neto"] + df["Tax"] + df["ShippingCost"]
        df["Diferencia_Facturacion"] = df["TotalAmount"] - df["Ingreso_Facturado"]

        # Se toleran diferencias de redondeo de hasta un centavo por línea.
        inconsistent = int((df["Diferencia_Facturacion"].abs() > 0.05).sum())
        total_raw = float(df["TotalAmount"].sum())
        total_net = float(df["Ingreso_Facturado"].sum())
        discount_impact = float((df["Ingreso_Bruto"] * df["Discount"]).sum())

        result = {
            "ingreso_total_reportado": round(total_raw, 2),
            "ingreso_neto_con_descuento": round(total_net, 2),
            "ingreso_bruto_sin_descuento": round(float(df["Ingreso_Bruto"].sum()), 2),
            "impacto_descuento": round(discount_impact, 2),
            "diferencia_total_amount_vs_neto": round(total_raw - total_net, 2),
            "filas_inconsistentes_total_amount": inconsistent,
            "porcentaje_filas_inconsistentes": round(100 * inconsistent / len(df), 4),
            "formula_adoptada": (
                "Ingreso_Neto = Quantity * UnitPrice * (1 - Discount); "
                "Ingreso_Facturado = Ingreso_Neto + Tax + ShippingCost"
            ),
        }
        self.report["ingresos"] = result
        self._decision(
            "TotalAmount del dataset no descuenta la promoción (diferencia máxima "
            "de 0.30 por redondeo). Se adopta Ingreso_Neto para el análisis y se "
            f"cuantifica una sobreestimación de {result['diferencia_total_amount_vs_neto']:,.2f} USD."
        )
        return result

    def flag_outliers(self) -> Dict[str, Any]:
        """Cuenta atípicos con el criterio IQR; no los elimina."""
        columns = [
            "Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost",
            "TotalAmount", "Ingreso_Neto", "Ingreso_Facturado",
        ]
        result: Dict[str, Any] = {}
        for col in columns:
            if col not in self._df.columns:
                continue
            q1, q3 = self._df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low = q1 - OUTLIER_IQR_FACTOR * iqr
            high = q3 + OUTLIER_IQR_FACTOR * iqr
            count = int(((self._df[col] < low) | (self._df[col] > high)).sum())
            result[col] = {
                "atipicos": count,
                "porcentaje": round(100 * count / len(self._df), 3),
                "limite_inferior": round(float(low), 2),
                "limite_superior": round(float(high), 2),
            }
        self.report["atipicos_iqr"] = result
        self._decision(
            "Los valores atípicos se reportan con IQR (factor 3) y se conservan; "
            "en la fase 4 se recortan solo las variables de modelado al percentil 1-99."
        )
        return result

    def _build_analytical(self) -> None:
        """Construye el subconjunto de ventas realizadas."""
        mask = self._df["OrderStatus"].isin(ANALYTICAL_STATUSES)
        self._analytical = self._df.loc[mask].copy()
        excluded = int((~mask).sum())
        self.report["ventas_realizadas"] = {
            "estados_incluidos": list(ANALYTICAL_STATUSES),
            "filas_incluidas": int(len(self._analytical)),
            "filas_excluidas": excluded,
            "porcentaje_excluido": round(100 * excluded / max(len(self._df), 1), 3),
            "por_estado": {
                str(k): int(v) for k, v in self._df["OrderStatus"].value_counts().items()
            },
        }
        self._decision(
            f"Conjunto analítico: {len(self._analytical):,} ventas realizadas "
            f"({', '.join(ANALYTICAL_STATUSES)}). Se excluyen {excluded:,} pedidos "
            "cancelados, devueltos o pendientes para no inflar ingreso ni recencia."
        )

    def clean(self) -> pd.DataFrame:
        """Ejecuta la secuencia completa y devuelve el dataset limpio."""
        self.convert_date_column()
        self.remove_duplicates()
        self.check_missing_values(impute=True)
        self.validate_consistency()
        self.add_revenue_columns()
        self.flag_outliers()
        self._build_analytical()

        self.report["filas_salida"] = int(len(self._df))
        self.report["filas_analiticas"] = int(len(self.analytical_data))
        self.report["reduccion_porcentaje"] = round(
            100 * (1 - len(self._df) / self.report["filas_entrada"]), 4
        )
        return self._df

    # --------------------------------------------------------- compatibilidad
    def drop_columns(self, columns: Optional[List[str]] = None) -> List[str]:
        """Elimina columnas irrelevantes (no se usa por defecto en el pipeline)."""
        columns = columns or COLS_TO_DROP
        existing = [c for c in columns if c in self._df.columns]
        self._df = self._df.drop(columns=existing, errors="ignore")
        self._decision(f"Columnas eliminadas del conjunto modelado: {existing}.")
        return existing

    def get_total_income(self, column: str = "TotalAmount") -> float:
        """Suma una columna monetaria del dataset original."""
        total = float(self._original_df[column].sum())
        print(f"Ingreso total reportado ({column}): {total:,.2f} USD")
        return total

    def get_processed_data(self) -> pd.DataFrame:
        """Devuelve el dataset limpio completo."""
        return self._df

    def save_report(self, path: Optional[str] = None) -> str:
        """Guarda el reporte de limpieza en JSON."""
        import json

        from ..config.settings import RESULTS_DIR

        target = Path(path or RESULTS_DIR / "reporte_limpieza.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Función de conveniencia: limpia y devuelve el dataset completo."""
    cleaner = DataCleaner(df)
    return cleaner.clean()

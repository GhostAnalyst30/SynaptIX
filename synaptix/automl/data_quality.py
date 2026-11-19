"""
Data Quality Engine: diagnóstico y corrección automática de calidad de datos.

Tres componentes:
1. DataQualityAnalyzer: detecta problemas
2. WarningSolver: propone/ejecuta correcciones (none, safe, aggressive)
3. DataQualityReport: registra resultados y recomendaciones
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .exceptions import AutoMLError


class SolverStrategy(Enum):
    """Estrategia de corrección de problemas de calidad."""
    NONE = "none"        # Solo analiza, no modifica
    SAFE = "safe"        # Correcciones de bajo riesgo
    AGGRESSIVE = "aggressive"  # Correcciones adicionales, requieren confirmación


class Severity(Enum):
    """Severidad del problema detectado."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ActionType(Enum):
    """Tipo de acción realizada."""
    NONE = "none"
    IMPUTE = "impute"
    DROP_COLUMN = "drop_column"
    DROP_ROWS = "drop_rows"
    CAP_OUTLIERS = "cap_outliers"
    RECODE_CATEGORIES = "recode_categories"
    CONVERT_DTYPE = "convert_dtype"
    FLAG_FOR_REVIEW = "flag_for_review"


@dataclass
class DataQualityIssue:
    """Problema de calidad detectado."""
    code: str
    message: str
    column: Optional[str] = None
    severity: Severity = Severity.WARNING
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Para el solver
    recommended_action: Optional[ActionType] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    requires_human_review: bool = False


@dataclass
class DataQualityAction:
    """Acción ejecutada por el solver."""
    issue_code: str
    column: Optional[str]
    action: ActionType
    params: Dict[str, Any]
    rows_affected: int = 0
    columns_affected: int = 0
    justification: str = ""
    requires_confirmation: bool = False


@dataclass
class DataQualityReport:
    """Reporte completo de calidad de datos."""
    issues: List[DataQualityIssue] = field(default_factory=list)
    actions_taken: List[DataQualityAction] = field(default_factory=list)
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    strategy: SolverStrategy = SolverStrategy.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issues": [
                {
                    "code": i.code,
                    "message": i.message,
                    "column": i.column,
                    "severity": i.severity.value,
                    "details": i.details,
                    "recommended_action": i.recommended_action.value if i.recommended_action else None,
                    "requires_human_review": i.requires_human_review,
                }
                for i in self.issues
            ],
            "actions_taken": [
                {
                    "issue_code": a.issue_code,
                    "column": a.column,
                    "action": a.action.value,
                    "params": a.params,
                    "rows_affected": a.rows_affected,
                    "columns_affected": a.columns_affected,
                    "justification": a.justification,
                    "requires_confirmation": a.requires_confirmation,
                }
                for a in self.actions_taken
            ],
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "strategy": self.strategy.value,
            "summary": {
                "total_issues": len(self.issues),
                "by_severity": self._count_by_severity(),
                "actions_executed": len(self.actions_taken),
            },
        }
    
    def _count_by_severity(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts
    
    def get_issues_dataframe(self) -> pd.DataFrame:
        """Issues como DataFrame para fácil inspección."""
        if not self.issues:
            return pd.DataFrame(columns=[
                "code", "message", "column", "severity", "recommended_action", "requires_review"
            ])
        return pd.DataFrame([{
            "code": i.code,
            "message": i.message,
            "column": i.column,
            "severity": i.severity.value,
            "recommended_action": i.recommended_action.value if i.recommended_action else None,
            "requires_review": i.requires_human_review,
        } for i in self.issues])
    
    def get_actions_dataframe(self) -> pd.DataFrame:
        """Acciones ejecutadas como DataFrame."""
        if not self.actions_taken:
            return pd.DataFrame(columns=[
                "issue_code", "column", "action", "rows_affected", "justification"
            ])
        return pd.DataFrame([{
            "issue_code": a.issue_code,
            "column": a.column,
            "action": a.action.value,
            "params": str(a.params),
            "rows_affected": a.rows_affected,
            "columns_affected": a.columns_affected,
            "justification": a.justification,
        } for a in self.actions_taken])


# ---------------------------------------------------------------------------
# DataQualityAnalyzer
# ---------------------------------------------------------------------------

# Umbrales configurables
DEFAULT_THRESHOLDS = {
    "high_missing": 0.5,
    "high_cardinality_abs": 50,
    "high_cardinality_rel": 0.5,
    "imbalance_ratio": 5.0,
    "rare_class_count": 10,
    "constant_threshold": 1.0,
    "quasi_constant": 0.99,
    "id_unique_ratio": 0.95,
    "outlier_iqr_factor": 1.5,
    "low_variance_threshold": 0.0,
    "skew_threshold": 10.0,
    "target_missing_threshold": 0.1,
    "train_test_distribution_shift": 0.1,
}


class DataQualityAnalyzer:
    """
    Analiza un DataFrame y detecta problemas de calidad.
    
    No modifica los datos, solo detecta y reporta.
    """
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        target: Optional[str] = None,
        task: Optional[str] = None,
        train_data: Optional[pd.DataFrame] = None,  # Para detectar shift train/test
    ):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.target = target
        self.task = task
        self.train_data = train_data
        self.issues: List[DataQualityIssue] = []
    
    def analyze(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Ejecuta todos los chequeos y devuelve lista de issues."""
        self.issues = []
        self._check_missing_values(df)
        self._check_infinite_values(df)
        self._check_constant_columns(df)
        self._check_duplicate_columns(df)
        self._check_duplicate_rows(df)
        self._check_high_cardinality(df)
        self._check_categorical_unknown_categories(df)
        self._check_class_imbalance(df)
        self._check_data_leakage(df)
        self._check_inconsistent_dtypes(df)
        self._check_outliers(df)
        self._check_low_variance(df)
        self._check_target_issues(df)
        self._check_train_test_shift(df)
        return self.issues
    
    def _add_issue(self, issue: DataQualityIssue) -> None:
        self.issues.append(issue)
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        n_rows = len(df)
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            pct_missing = n_missing / n_rows
            if pct_missing >= self.thresholds["high_missing"]:
                self._add_issue(DataQualityIssue(
                    code="high_missing",
                    message=f"{pct_missing:.1%} de valores faltantes",
                    column=col,
                    severity=Severity.WARNING,
                    details={"n_missing": int(n_missing), "pct_missing": pct_missing},
                    recommended_action=ActionType.IMPUTE,
                    action_params={"strategy": "median" if ptypes.is_numeric_dtype(df[col]) else "most_frequent"},
                ))
            elif n_missing > 0:
                self._add_issue(DataQualityIssue(
                    code="missing_values",
                    message=f"{n_missing} valores faltantes ({pct_missing:.1%})",
                    column=col,
                    severity=Severity.INFO,
                    details={"n_missing": int(n_missing), "pct_missing": pct_missing},
                    recommended_action=ActionType.IMPUTE,
                    action_params={"strategy": "median" if ptypes.is_numeric_dtype(df[col]) else "most_frequent"},
                ))
    
    def _check_infinite_values(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = df[col].to_numpy()
            n_inf = np.sum(np.isinf(vals))
            if n_inf > 0:
                self._add_issue(DataQualityIssue(
                    code="infinite_values",
                    message=f"{n_inf} valores infinitos",
                    column=col,
                    severity=Severity.WARNING,
                    details={"n_inf": int(n_inf)},
                    recommended_action=ActionType.IMPUTE,
                    action_params={"strategy": "median", "replace_inf": True},
                ))
    
    def _check_constant_columns(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                self._add_issue(DataQualityIssue(
                    code="constant_column",
                    message="Columna constante: no aporta información",
                    column=col,
                    severity=Severity.WARNING,
                    details={"n_unique": int(n_unique)},
                    recommended_action=ActionType.DROP_COLUMN,
                    action_params={},
                    requires_human_review=True,
                ))
    
    def _check_duplicate_columns(self, df: pd.DataFrame) -> None:
        # Comparar columnas dos a dos (puede ser lento para muchas columnas)
        cols = list(df.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                if df[c1].equals(df[c2]):
                    self._add_issue(DataQualityIssue(
                        code="duplicate_column",
                        message=f"Columna duplicada de '{c1}'",
                        column=c2,
                        severity=Severity.WARNING,
                        details={"duplicate_of": c1},
                        recommended_action=ActionType.DROP_COLUMN,
                        action_params={},
                        requires_human_review=True,
                    ))
    
    def _check_duplicate_rows(self, df: pd.DataFrame) -> None:
        n_dup = df.duplicated().sum()
        if n_dup > 0:
            self._add_issue(DataQualityIssue(
                code="duplicate_rows",
                message=f"{n_dup} filas duplicadas",
                column=None,
                severity=Severity.INFO,
                details={"n_duplicates": int(n_dup)},
                recommended_action=ActionType.DROP_ROWS,
                action_params={"keep": "first"},
            ))
    
    def _check_high_cardinality(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if ptypes.is_numeric_dtype(df[col]):
                continue
            n_unique = df[col].nunique(dropna=True)
            non_null = df[col].notna().sum()
            if n_unique > self.thresholds["high_cardinality_abs"] or \
               (non_null > 0 and n_unique / non_null > self.thresholds["high_cardinality_rel"]):
                self._add_issue(DataQualityIssue(
                    code="high_cardinality",
                    message=f"{n_unique} categorías únicas (alto cardinalidad)",
                    column=col,
                    severity=Severity.INFO,
                    details={"n_unique": int(n_unique), "non_null": int(non_null)},
                    recommended_action=ActionType.RECODE_CATEGORIES,
                    action_params={"max_categories": 20, "group_rare": "infrequent"},
                ))
    
    def _check_categorical_unknown_categories(self, df: pd.DataFrame) -> None:
        # Este check es más relevante en predict, pero podemos detectar
        # categorías muy raras que podrían causar problemas
        for col in df.select_dtypes(include=["object", "category"]).columns:
            vc = df[col].value_counts(dropna=True)
            rare = vc[vc < self.thresholds["rare_class_count"]]
            if len(rare) > 0:
                self._add_issue(DataQualityIssue(
                    code="rare_categories",
                    message=f"{len(rare)} categorías con menos de {self.thresholds['rare_class_count']} ejemplos",
                    column=col,
                    severity=Severity.INFO,
                    details={"rare_categories": rare.index.tolist()[:10], "n_rare": len(rare)},
                    recommended_action=ActionType.RECODE_CATEGORIES,
                    action_params={"min_frequency": self.thresholds["rare_class_count"]},
                ))
    
    def _check_class_imbalance(self, df: pd.DataFrame) -> None:
        if self.target is None or self.target not in df.columns:
            return
        if self.task != "classification":
            return
        y = df[self.target].dropna()
        if len(y) == 0:
            return
        counts = y.value_counts()
        if len(counts) < 2:
            return
        ratio = counts.iloc[0] / counts.iloc[-1]
        if ratio >= self.thresholds["imbalance_ratio"]:
            self._add_issue(DataQualityIssue(
                code="class_imbalance",
                message=f"Desbalance de clases {ratio:.1f}:1",
                column=self.target,
                severity=Severity.WARNING,
                details={"imbalance_ratio": ratio, "class_counts": counts.to_dict()},
                recommended_action=ActionType.FLAG_FOR_REVIEW,
                action_params={"suggest_scoring": "balanced_accuracy"},
                requires_human_review=True,
            ))
    
    def _check_data_leakage(self, df: pd.DataFrame) -> None:
        if self.target is None or self.target not in df.columns:
            return
        target_lower = str(self.target).lower()
        for col in df.columns:
            if col == self.target:
                continue
            if target_lower in str(col).lower() and len(target_lower) >= 3:
                self._add_issue(DataQualityIssue(
                    code="potential_leakage_name",
                    message=f"Nombre contiene el del target ('{self.target}'): posible fuga",
                    column=col,
                    severity=Severity.WARNING,
                    details={"target_name": self.target},
                    recommended_action=ActionType.FLAG_FOR_REVIEW,
                    action_params={},
                    requires_human_review=True,
                ))
    
    def _check_inconsistent_dtypes(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if df[col].dtype == "object":
                # Verificar si parece numérico pero es object
                sample = df[col].dropna()
                if len(sample) > 0:
                    try:
                        pd.to_numeric(sample, errors="raise")
                        self._add_issue(DataQualityIssue(
                            code="numeric_as_object",
                            message="Columna numérica almacenada como object",
                            column=col,
                            severity=Severity.INFO,
                            details={"sample_values": sample.head(5).tolist()},
                            recommended_action=ActionType.CONVERT_DTYPE,
                            action_params={"target_dtype": "float"},
                        ))
                    except (ValueError, TypeError):
                        pass
    
    def _check_outliers(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == self.target:
                continue
            vals = df[col].dropna()
            if len(vals) < 10:
                continue
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            factor = self.thresholds["outlier_iqr_factor"]
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            n_outliers = ((vals < lower) | (vals > upper)).sum()
            if n_outliers > 0 and n_outliers / len(vals) > 0.01:
                self._add_issue(DataQualityIssue(
                    code="outliers_iqr",
                    message=f"{n_outliers} outliers detectados (IQR factor={factor})",
                    column=col,
                    severity=Severity.INFO,
                    details={"n_outliers": int(n_outliers), "pct": n_outliers / len(vals), "bounds": (float(lower), float(upper))},
                    recommended_action=ActionType.CAP_OUTLIERS,
                    action_params={"lower": float(lower), "upper": float(upper)},
                ))
    
    def _check_low_variance(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == self.target:
                continue
            vals = df[col].dropna()
            if len(vals) < 2:
                continue
            var = vals.var()
            if var <= self.thresholds["low_variance_threshold"]:
                self._add_issue(DataQualityIssue(
                    code="low_variance",
                    message=f"Varianza muy baja o cero ({var:.6f})",
                    column=col,
                    severity=Severity.INFO,
                    details={"variance": float(var)},
                    recommended_action=ActionType.DROP_COLUMN,
                    action_params={},
                    requires_human_review=True,
                ))
    
    def _check_target_issues(self, df: pd.DataFrame) -> None:
        if self.target is None or self.target not in df.columns:
            return
        y = df[self.target]
        n_missing = y.isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df)
            severity = Severity.WARNING if pct > self.thresholds["target_missing_threshold"] else Severity.INFO
            self._add_issue(DataQualityIssue(
                code="target_missing",
                message=f"{n_missing} valores faltantes en target ({pct:.1%})",
                column=self.target,
                severity=severity,
                details={"n_missing": int(n_missing), "pct_missing": pct},
                recommended_action=ActionType.DROP_ROWS,
                action_params={"subset": [self.target]},
            ))
        
        if self.task == "regression":
            vals = pd.to_numeric(y, errors="coerce").dropna()
            if len(vals) > 2:
                skew = vals.skew()
                if abs(skew) > self.thresholds["skew_threshold"]:
                    self._add_issue(DataQualityIssue(
                        code="target_skewed",
                        message=f"Target muy sesgado (skew={skew:.2f})",
                        column=self.target,
                        severity=Severity.INFO,
                        details={"skew": float(skew)},
                        recommended_action=ActionType.FLAG_FOR_REVIEW,
                        action_params={"suggest_transform": "log" if (vals > 0).all() else "yeo-johnson"},
                    ))
    
    def _check_train_test_shift(self, df: pd.DataFrame) -> None:
        if self.train_data is None:
            return
        # Comparar distribuciones de columnas numéricas
        common_cols = set(df.select_dtypes(include=[np.number]).columns) & \
                      set(self.train_data.select_dtypes(include=[np.number]).columns)
        for col in common_cols:
            if col == self.target:
                continue
            train_vals = self.train_data[col].dropna()
            test_vals = df[col].dropna()
            if len(train_vals) < 10 or len(test_vals) < 10:
                continue
            # KS test aproximado: diferencia en medias normalizada
            train_mean, test_mean = train_vals.mean(), test_vals.mean()
            train_std = train_vals.std()
            if train_std > 0:
                shift = abs(test_mean - train_mean) / train_std
                if shift > self.thresholds["train_test_distribution_shift"]:
                    self._add_issue(DataQualityIssue(
                        code="distribution_shift",
                        message=f"Posible shift de distribución (shift={shift:.2f} stds)",
                        column=col,
                        severity=Severity.WARNING,
                        details={"shift_stds": shift, "train_mean": float(train_mean), "test_mean": float(test_mean)},
                        recommended_action=ActionType.FLAG_FOR_REVIEW,
                        action_params={},
                        requires_human_review=True,
                    ))


# ---------------------------------------------------------------------------
# WarningSolver
# ---------------------------------------------------------------------------

# Correcciones que eliminan filas: son deterministas (no estiman parámetros con
# los datos) y deben aplicarse antes de separar el holdout para no desalinear
# los índices de entrenamiento y prueba.
ROW_LEVEL_CODES: frozenset = frozenset({"duplicate_rows", "target_missing"})

class WarningSolver:
    """
    Aplica correcciones a los problemas detectados según la estrategia.
    
    Estrategias:
    - NONE: Solo analiza, no modifica
    - SAFE: Correcciones de bajo riesgo (imputación, infinitos, constantes)
    - AGGRESSIVE: Incluye SAFE + outliers, categorías raras, conversión de tipos
    """
    
    def __init__(
        self,
        strategy: SolverStrategy = SolverStrategy.NONE,
        require_confirmation: bool = True,
        custom_actions: Optional[Dict[str, Callable]] = None,
    ):
        self.strategy = strategy
        self.require_confirmation = require_confirmation
        self.custom_actions = custom_actions or {}
        self.actions_taken: List[DataQualityAction] = []
    
    def solve(
        self,
        df: pd.DataFrame,
        issues: List[DataQualityIssue],
        confirm_callback: Optional[Callable[[str], bool]] = None,
        reference: Optional[pd.DataFrame] = None,
        codes: Optional[Any] = None,
        exclude_codes: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, List[DataQualityAction]]:
        """
        Aplica correcciones según la estrategia.
        
        Parameters
        ----------
        df : DataFrame a corregir (se modifica una copia)
        issues : Lista de issues detectados
        confirm_callback : Función para confirmar acciones agresivas.
                          Recibe descripción y devuelve True/False.
        reference : DataFrame, optional
            Filas de referencia para estimar parámetros (media, mediana,
            cuantiles, frecuencias). Debe ser el conjunto de entrenamiento:
            estimar con ``df`` completo filtraría información del test.
        codes : iterable de str, optional
            Si se indica, solo se resuelven esos códigos de problema.
        exclude_codes : iterable de str, optional
            Códigos que no se resuelven en esta llamada.
        
        Returns
        -------
        (df_corregido, acciones_tomadas)
        """
        if self.strategy == SolverStrategy.NONE:
            return df, []
        reference = df if reference is None else reference
        only = None if codes is None else set(codes)
        skip = set() if exclude_codes is None else set(exclude_codes)
        
        df = df.copy()
        self.actions_taken = []
        
        # Mapear issue_code -> acción
        safe_actions = {
            "missing_values": self._impute_missing,
            "high_missing": self._impute_missing,
            "infinite_values": self._handle_infinite,
            "constant_column": self._drop_constant,
            "duplicate_rows": self._drop_duplicate_rows,
            "target_missing": self._drop_target_missing,
        }
        
        aggressive_actions = {
            **safe_actions,
            "outliers_iqr": self._cap_outliers,
            "rare_categories": self._group_rare_categories,
            "numeric_as_object": self._convert_numeric_object,
            "high_cardinality": self._group_rare_categories,
            "duplicate_column": self._drop_duplicate_column,
            "low_variance": self._drop_low_variance,
        }
        
        action_map = aggressive_actions if self.strategy == SolverStrategy.AGGRESSIVE else safe_actions
        
        for issue in issues:
            if issue.code not in action_map:
                continue
            if only is not None and issue.code not in only:
                continue
            if issue.code in skip:
                continue
            
            # Verificar si requiere confirmación
            if issue.requires_human_review and self.require_confirmation:
                if confirm_callback is None:
                    # Por defecto, no ejecutar acciones que requieren confirmación
                    self.actions_taken.append(DataQualityAction(
                        issue_code=issue.code,
                        column=issue.column,
                        action=ActionType.NONE,
                        params={},
                        justification=f"Acción requiere confirmación humana (no se proveyó callback)",
                        requires_confirmation=True,
                    ))
                    continue
                desc = f"{issue.code} en '{issue.column}': {issue.message}. Acción: {issue.recommended_action.value}"
                if not confirm_callback(desc):
                    self.actions_taken.append(DataQualityAction(
                        issue_code=issue.code,
                        column=issue.column,
                        action=ActionType.NONE,
                        params={},
                        justification="Usuario rechazó la acción",
                        requires_confirmation=True,
                    ))
                    continue
            
            # Ejecutar acción
            try:
                action_func = action_map[issue.code]
                rows_before = len(df)
                cols_before = len(df.columns)
                df = action_func(df, issue, reference)
                rows_after = len(df)
                cols_after = len(df.columns)
                
                self.actions_taken.append(DataQualityAction(
                    issue_code=issue.code,
                    column=issue.column,
                    action=issue.recommended_action or ActionType.NONE,
                    params=issue.action_params,
                    rows_affected=rows_before - rows_after,
                    columns_affected=cols_before - cols_after,
                    justification=f"Ejecutado automáticamente (estrategia: {self.strategy.value})",
                    requires_confirmation=issue.requires_human_review,
                ))
            except Exception as exc:
                self.actions_taken.append(DataQualityAction(
                    issue_code=issue.code,
                    column=issue.column,
                    action=ActionType.NONE,
                    params={},
                    justification=f"Error ejecutando acción: {exc}",
                    requires_confirmation=False,
                ))
        
        return df, self.actions_taken
    
    # --- Acciones SAFE ---
    
    def _impute_missing(self, df: pd.DataFrame, issue: DataQualityIssue,
                        reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is None or col not in df.columns:
            return df
        source = df if reference is None or col not in reference.columns else reference
        strategy = issue.action_params.get("strategy", "median")
        numeric = ptypes.is_numeric_dtype(source[col])
        fill_val = None
        if strategy == "median" and numeric:
            fill_val = source[col].median()
        elif strategy == "mean" and numeric:
            fill_val = source[col].mean()
        elif strategy == "most_frequent":
            mode = source[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else None
        if fill_val is None or (isinstance(fill_val, float) and not np.isfinite(fill_val)):
            fill_val = 0 if ptypes.is_numeric_dtype(df[col]) else ""
        df[col] = df[col].fillna(fill_val)
        return df
    
    def _handle_infinite(self, df: pd.DataFrame, issue: DataQualityIssue,
                         reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is None or col not in df.columns:
            return df
        if issue.action_params.get("replace_inf", True):
            # Reemplazar inf con NaN y luego imputar con la mediana de train
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            source = df if reference is None or col not in reference.columns else reference
            finite = source[col].replace([np.inf, -np.inf], np.nan).dropna()
            fill_val = float(finite.median()) if len(finite) else 0.0
            df[col] = df[col].fillna(fill_val)
        return df
    
    def _drop_constant(self, df: pd.DataFrame, issue: DataQualityIssue,
                       reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is not None and col in df.columns:
            df = df.drop(columns=[col])
        return df
    
    def _drop_duplicate_rows(self, df: pd.DataFrame, issue: DataQualityIssue,
                             reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        keep = issue.action_params.get("keep", "first")
        df = df.drop_duplicates(keep=keep)
        return df
    
    def _drop_target_missing(self, df: pd.DataFrame, issue: DataQualityIssue,
                             reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is not None and col in df.columns:
            df = df.dropna(subset=[col])
        return df
    
    # --- Acciones AGGRESSIVE ---
    
    def _cap_outliers(self, df: pd.DataFrame, issue: DataQualityIssue,
                      reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is None or col not in df.columns:
            return df
        lower = issue.action_params.get("lower")
        upper = issue.action_params.get("upper")
        if lower is not None and upper is not None:
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df
    
    def _group_rare_categories(self, df: pd.DataFrame, issue: DataQualityIssue,
                               reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is None or col not in df.columns:
            return df
        min_freq = issue.action_params.get("min_frequency", 10)
        max_cats = issue.action_params.get("max_categories", 20)
        source = df if reference is None or col not in reference.columns else reference
        vc = source[col].value_counts(dropna=True)
        # Agrupar raras
        rare = vc[vc < min_freq].index
        if len(rare) > 0:
            df[col] = df[col].where(~df[col].isin(rare), "infrequent")
        # Si aún hay muchas, agrupar las menos frecuentes
        vc2 = df[col].value_counts(dropna=True)
        if len(vc2) > max_cats:
            to_group = vc2.iloc[max_cats-1:].index  # mantener top max_cats-1 + infrequent
            df[col] = df[col].where(~df[col].isin(to_group), "infrequent")
        return df
    
    def _convert_numeric_object(self, df: pd.DataFrame, issue: DataQualityIssue,
                                reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is not None and col in df.columns:
            target_dtype = issue.action_params.get("target_dtype", "float")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(target_dtype)
        return df
    
    def _drop_duplicate_column(self, df: pd.DataFrame, issue: DataQualityIssue,
                               reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is not None and col in df.columns:
            df = df.drop(columns=[col])
        return df
    
    def _drop_low_variance(self, df: pd.DataFrame, issue: DataQualityIssue,
                           reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        col = issue.column
        if col is not None and col in df.columns:
            df = df.drop(columns=[col])
        return df


# ---------------------------------------------------------------------------
# Función de alto nivel para integración en AutoPipeline
# ---------------------------------------------------------------------------

def analyze_and_solve_data_quality(
    df: pd.DataFrame,
    target: Optional[str] = None,
    task: Optional[str] = None,
    strategy: Union[str, SolverStrategy] = "safe",
    thresholds: Optional[Dict[str, float]] = None,
    train_data: Optional[pd.DataFrame] = None,
    confirm_callback: Optional[Callable[[str], bool]] = None,
    reference: Optional[pd.DataFrame] = None,
    codes: Optional[Any] = None,
    exclude_codes: Optional[Any] = None,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Analiza calidad de datos y aplica correcciones según estrategia.
    
    Parameters
    ----------
    df : DataFrame a analizar y corregir
    target : Nombre de la columna target
    task : "classification" o "regression"
    strategy : "none", "safe", o "aggressive"
    thresholds : Umbrales personalizados
    train_data : Datos de entrenamiento (para detectar shift en test)
    confirm_callback : Callback para confirmar acciones agresivas
    reference : DataFrame, optional
        Filas usadas para *detectar* problemas y estimar los parámetros de las
        correcciones (medianas, cuantiles, frecuencias). Debe ser el conjunto de
        entrenamiento; si es None se analiza ``df`` completo.
    codes, exclude_codes : iterable de str, optional
        Restringen qué códigos de problema se resuelven en esta llamada.
    
    Returns
    -------
    (df_corregido, reporte)
    """
    if isinstance(strategy, str):
        strategy = SolverStrategy(strategy)
    
    analysis_frame = df if reference is None else reference
    analyzer = DataQualityAnalyzer(
        thresholds=thresholds,
        target=target,
        task=task,
        train_data=train_data,
    )
    issues = analyzer.analyze(analysis_frame)
    
    solver = WarningSolver(
        strategy=strategy,
        require_confirmation=True,
    )
    df_fixed, actions = solver.solve(df, issues, confirm_callback,
                                     reference=reference, codes=codes,
                                     exclude_codes=exclude_codes)
    
    report = DataQualityReport(
        issues=issues,
        actions_taken=actions,
        original_shape=df.shape,
        final_shape=df_fixed.shape,
        strategy=strategy,
    )
    
    return df_fixed, report
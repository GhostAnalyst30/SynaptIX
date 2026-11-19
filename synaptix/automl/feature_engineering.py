"""
Feature Engineering Engine: transformaciones de variables derivadas persistentes.

Permite definir variables derivadas mediante expresiones que se evalúan
tanto en entrenamiento como en inferencia, garantizando consistencia.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .exceptions import AutoMLError


# Operadores que pueden producir una división por cero
_DIVISION_OPERATORS: tuple = (ast.Div, ast.FloorDiv, ast.Mod)

# Operadores permitidos para evaluación segura de expresiones
_ALLOWED_OPERATORS: Dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Funciones permitidas
_ALLOWED_FUNCTIONS: Dict[str, Callable] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "log": np.log,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "where": np.where,
    "clip": np.clip,
    "isnan": np.isnan,
    "isinf": np.isinf,
}

# Constantes permitidas
_ALLOWED_CONSTANTS: Dict[str, Any] = {
    "pi": np.pi,
    "e": np.e,
    "inf": np.inf,
    "nan": np.nan,
}


class ExpressionEvaluator(ast.NodeVisitor):
    """Evaluador seguro de expresiones AST sin usar eval().

    Parameters
    ----------
    columns : dict
        Nombre de columna -> valores numéricos.
    div_zero : {"error", "inf", "nan", "zero"}, default="nan"
        Política de división por cero. NumPy no lanza ``ZeroDivisionError``
        (devuelve inf/nan), así que la política se aplica aquí y no con un
        ``except``: de lo contrario ``"zero"`` y ``"error"`` no tendrían efecto.
    """

    def __init__(self, columns: Dict[str, np.ndarray], div_zero: str = "nan"):
        self.columns = columns
        self.div_zero = div_zero
        self._used_columns: Set[str] = set()

    def visit(self, node: ast.AST) -> Any:
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise AutoMLError(
            f"Expresión no permitida: {ast.dump(node)[:100]}. "
            "Solo se permiten operaciones aritméticas básicas y funciones listadas."
        )

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise AutoMLError(f"Operador no permitido: {op_type.__name__}")
        is_division = op_type in _DIVISION_OPERATORS
        if is_division:
            left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = _ALLOWED_OPERATORS[op_type](left, right)
        if is_division:
            result = self._resolve_division(result, right)
        return result

    def _resolve_division(self, result: Any, denominator: Any) -> Any:
        """Aplica ``div_zero`` a los elementos cuyo denominador es cero."""
        zero = np.equal(denominator, 0)
        if not np.any(zero):
            return result
        if self.div_zero == "error":
            raise AutoMLError(
                "División por cero en una variable derivada; usa "
                "fe_handle_div_zero='nan', 'inf' o 'zero' para permitirla."
            )
        result = np.asarray(result, dtype=float)
        if self.div_zero == "zero":
            return np.where(zero, 0.0, result)
        if self.div_zero == "inf":
            with np.errstate(invalid="ignore"):
                filled = np.where(
                    np.isnan(result),
                    np.nan,
                    np.where(np.signbit(result), -np.inf, np.inf),
                )
            return np.where(zero, filled, result)
        return np.where(zero, np.nan, result)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise AutoMLError(f"Operador unario no permitido: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise AutoMLError("Solo se permiten llamadas a funciones simples")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise AutoMLError(
                f"Función '{func_name}' no permitida. "
                f"Permitidas: {sorted(_ALLOWED_FUNCTIONS)}"
            )
        func = _ALLOWED_FUNCTIONS[func_name]
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        if node.id not in self.columns:
            raise AutoMLError(f"Columna '{node.id}' no existe en los datos")
        self._used_columns.add(node.id)
        return self.columns[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise AutoMLError("Indexación/slicing no permitido en expresiones")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise AutoMLError("Acceso a atributos no permitido en expresiones")

    def get_used_columns(self) -> Set[str]:
        return self._used_columns


def parse_expression(expr: str) -> ast.Expression:
    """Parsea una expresión a AST validando sintaxis."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise AutoMLError(f"Expresión inválida '{expr}': {exc.msg}") from exc
    return tree


def evaluate_expression(
    tree: ast.Expression,
    columns: Dict[str, np.ndarray],
    div_zero: str = "nan",
) -> tuple[np.ndarray, Set[str]]:
    """Evalúa un AST de expresión con las columnas dadas."""
    evaluator = ExpressionEvaluator(columns, div_zero=div_zero)
    result = evaluator.visit(tree)
    if not isinstance(result, np.ndarray):
        result = np.full(len(next(iter(columns.values()))), result, dtype=float)
    return result, evaluator.get_used_columns()


@dataclass
class DerivedFeature:
    """Definición de una feature derivada."""
    name: str
    expression: str
    ast_tree: ast.Expression
    depends_on: Set[str] = field(default_factory=set)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que crea variables derivadas a partir de expresiones.

    Las expresiones se parsean una vez en fit() y se reevalúan en transform()
    con las mismas reglas, garantizando consistencia train/inference.

    Parameters
    ----------
    associations : dict[str, str]
        Mapeo nombre_nueva_columna -> expresión (ej: "ratio": "a / b")
    handle_division_by_zero : {"error", "inf", "nan", "zero"}, default="nan"
        Qué hacer ante división por cero: error, inf, nan, o 0.
    handle_invalid : {"error", "nan"}, default="nan"
        Qué hacer ante resultados inválidos (inf, nan).
    """

    def __init__(
        self,
        associations: Optional[Dict[str, str]] = None,
        handle_division_by_zero: str = "nan",
        handle_invalid: str = "nan",
    ):
        self.associations = associations or {}
        self.handle_division_by_zero = handle_division_by_zero
        self.handle_invalid = handle_invalid
        self.derived_features_: List[DerivedFeature] = []
        self.input_columns_: List[str] = []
        self.output_columns_: List[str] = []

    def _validate_params(self) -> None:
        if self.handle_division_by_zero not in ("error", "inf", "nan", "zero"):
            raise AutoMLError(
                f"handle_division_by_zero debe ser uno de "
                f"('error', 'inf', 'nan', 'zero'), recibido: {self.handle_division_by_zero!r}"
            )
        if self.handle_invalid not in ("error", "nan"):
            raise AutoMLError(
                f"handle_invalid debe ser 'error' o 'nan', recibido: {self.handle_invalid!r}"
            )
        for name, expr in self.associations.items():
            if not isinstance(name, str) or not name:
                raise AutoMLError("Los nombres de variables derivadas deben ser strings no vacíos")
            if not isinstance(expr, str) or not expr.strip():
                raise AutoMLError(f"Expresión para '{name}' debe ser string no vacío")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineeringTransformer":
        """Parsea y valida expresiones, detecta dependencias."""
        self._validate_params()
        self.input_columns_ = list(X.columns)
        self.derived_features_ = []
        self.output_columns_ = list(X.columns)

        # Parsear todas las expresiones primero
        parsed_trees: Dict[str, ast.Expression] = {}
        for name, expr in self.associations.items():
            if name in self.output_columns_:
                raise AutoMLError(
                    f"El nombre '{name}' ya existe en los datos de entrada. "
                    "Usa un nombre diferente para la variable derivada."
                )
            tree = parse_expression(expr)
            parsed_trees[name] = tree

        # Procesar en orden, agregando cada columna derivada a las disponibles
        # Empezar con columnas numéricas de entrada
        available_arrays = {
            col: X[col].to_numpy(dtype=float)
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        }

        for name, expr in self.associations.items():
            tree = parsed_trees[name]

            # Validar dependencias con un evaluador dummy
            evaluator = ExpressionEvaluator(available_arrays, self.handle_division_by_zero)
            try:
                evaluator.visit(tree)
            except AutoMLError:
                raise
            except Exception as exc:
                raise AutoMLError(f"Error analizando expresión para '{name}': {expr}") from exc
            used = evaluator.get_used_columns()

            # Verificar que las columnas usadas son numéricas y están disponibles
            for col in used:
                if col not in available_arrays:
                    raise AutoMLError(
                        f"La columna '{col}' usada en '{name} = {expr}' "
                        f"no está disponible o no es numérica. "
                        f"Columnas disponibles: {sorted(available_arrays.keys())}"
                    )

            # Verificar dependencias circulares
            if name in used:
                raise AutoMLError(
                    f"Dependencia circular: '{name}' se referencia a sí misma en '{expr}'"
                )

            # Evaluar realmente para validar que produce valores válidos
            try:
                values, actual_used = evaluate_expression(
                    tree, available_arrays, self.handle_division_by_zero)
            except AutoMLError:
                raise
            except Exception as exc:
                raise AutoMLError(f"Error evaluando expresión para '{name}': {expr}") from exc

            derived = DerivedFeature(
                name=name,
                expression=expr,
                ast_tree=tree,
                depends_on=used,
            )
            self.derived_features_.append(derived)
            self.output_columns_.append(name)

            # Agregar los valores calculados para las siguientes expresiones
            # (con ceros, una expresión encadenada que divida por esta columna
            # fallaría en fit aunque los datos reales no tengan ceros).
            available_arrays[name] = np.broadcast_to(
                np.asarray(values, dtype=float), (len(X),))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica las expresiones parseadas a nuevos datos."""
        if not self.derived_features_:
            return X

        # Validar columnas de entrada (solo las que se necesitan para expresiones, EXCLUYENDO las derivadas)
        base_columns = set()
        for derived in self.derived_features_:
            base_columns.update(derived.depends_on)
        # Eliminar columnas que son derivadas de este mismo transformer
        derived_names = {d.name for d in self.derived_features_}
        required_input_columns = base_columns - derived_names

        missing = [c for c in required_input_columns if c not in X.columns]
        if missing:
            raise AutoMLError(
                f"Faltan columnas base requeridas para feature engineering: {missing}. "
                f"Columnas necesarias: {sorted(required_input_columns)}"
            )

        # Preparar arrays solo de columnas numéricas base necesarias
        col_arrays = {}
        for col in required_input_columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise AutoMLError(
                    f"La columna '{col}' no es numérica y es requerida para feature engineering."
                )
            col_arrays[col] = X[col].to_numpy(dtype=float)

        result = X.copy()

        for derived in self.derived_features_:
            # Verificar que las dependencias están disponibles (ya sea en input o ya calculadas)
            missing_deps = [c for c in derived.depends_on if c not in col_arrays]
            if missing_deps:
                raise AutoMLError(
                    f"Faltan dependencias para '{derived.name}': {missing_deps}. "
                    f"Expresión: {derived.expression}"
                )

            try:
                values, _ = evaluate_expression(
                    derived.ast_tree, col_arrays, self.handle_division_by_zero)
            except AutoMLError:
                raise
            except Exception as exc:
                raise AutoMLError(
                    f"Error evaluando '{derived.name} = {derived.expression}': {exc}"
                ) from exc

            # Manejar valores inválidos (inf, nan). Con div_zero="inf" los
            # infinitos son intencionales y se conservan.
            if self.handle_division_by_zero == "inf":
                values = np.asarray(values, dtype=float)
            elif self.handle_invalid == "error":
                if not np.all(np.isfinite(values)):
                    raise AutoMLError(
                        f"Expresión '{derived.expression}' produjo valores inválidos (inf/nan)"
                    )
            else:  # "nan"
                values = np.where(np.isfinite(values), values, np.nan)

            result[derived.name] = values
            col_arrays[derived.name] = values  # Disponible para siguientes expresiones

        return result

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.array(self.output_columns_, dtype=object)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "associations": self.associations,
            "handle_division_by_zero": self.handle_division_by_zero,
            "handle_invalid": self.handle_invalid,
        }

    def set_params(self, **params) -> "FeatureEngineeringTransformer":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        if not self.associations:
            return "FeatureEngineeringTransformer()"
        exprs = ", ".join(f"{k}={v!r}" for k, v in self.associations.items())
        return f"FeatureEngineeringTransformer({{{exprs}}})"
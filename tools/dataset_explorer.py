import os
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd
from langchain_core.tools import tool


@tool
def analyze_dataset(input_str: str) -> str:
    """
    Analyze tabular data from a file with optional query.
    Requires Pandas library.

    Args:
        input_str: A string of the form "<file_path>||<query>" or
        just "<file_path>".

    Returns:
        Analysis results or an error message.
    """
    try:
        tab_analyzer = TabularDataAnalyzer()
        if "||" in input_str:
            file_path, query = input_str.split("||", 1)
        else:
            file_path, query = input_str, None
        result = tab_analyzer.analyze(file_path.strip(), query)
        return str(result)
    except Exception as e:
        return f"Tabular analysis failed: {str(e)}"


class FileType(Enum):
    CSV = "csv"
    EXCEL = "excel"
    UNKNOWN = "unknown"


class TabularDataAnalyzer:
    """
    Advanced tabular data analysis tool with:
    - Automatic file type detection
    - Rich statistical summaries
    - Context-aware output formatting
    - Error handling and validation

    Usage:
        analyzer = TabularDataAnalyzer()
        result = analyzer.analyze("data.csv")
        print(result["summary"])
    """

    def __init__(self):
        self.required_packages = {
            FileType.CSV: ["pandas"],
            FileType.EXCEL: ["pandas", "openpyxl"],
        }

    def analyze(self, file_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze tabular data file with automatic format detection.

        Args:
            file_path: Path to CSV/Excel file
            query: Optional question about the data

        Returns:
            Dictionary containing:
            - summary: Human-readable analysis
            - stats: Detailed statistics
            - metadata: File information
            - error: None or error message
        """
        result = {"summary": "", "stats": {}, "metadata": {}, "error": None}

        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Detect file type
            file_type = self._detect_file_type(file_path)
            if file_type == FileType.UNKNOWN:
                raise ValueError("Unsupported file format")

            # Check dependencies
            self._check_dependencies(file_type)

            # Read file
            df = self._read_file(file_path, file_type)

            # Generate results
            result["metadata"] = self._get_metadata(df, file_type)
            result["stats"] = self._get_statistics(df)
            result["summary"] = self._generate_summary(
                result["metadata"], result["stats"], query
            )

        except Exception as e:
            result["error"] = str(e)

        return result

    def _detect_file_type(self, file_path: str) -> FileType:
        """Determine file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return FileType.CSV
        elif ext in (".xlsx", ".xls"):
            return FileType.EXCEL
        return FileType.UNKNOWN

    def _check_dependencies(self, file_type: FileType):
        """Verify required packages are installed"""
        for package in self.required_packages[file_type]:
            try:
                __import__(package)
            except ImportError:
                raise ImportError(
                    f"Required package not found: {package}. "
                    f"Install with 'pip install {' '.join(self.required_packages[file_type])}'"
                )

    def _read_file(self, file_path: str, file_type: FileType) -> pd.DataFrame:
        """Read file based on detected type"""
        if file_type == FileType.CSV:
            return pd.read_csv(file_path)
        elif file_type == FileType.EXCEL:
            return pd.read_excel(file_path, engine="openpyxl")
        raise ValueError("Unsupported file format")

    def _get_metadata(self, df: pd.DataFrame, file_type: FileType) -> Dict[str, Any]:
        """Extract file and data structure metadata"""
        return {
            "file_type": file_type.value,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isna().sum().to_dict(),
        }

    def _get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        stats = {}
        desc = df.describe(include="all", percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])

        # Numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            stats[col] = {
                "10th_percentile": f"10% of values are below {desc[col]['10%']}",
                "25th_percentile": f"25% of values are below {desc[col]['25%']}",
                "50th_percentile": f"Median (50th percentile) value is {desc[col]['50%']}",
                "75th_percentile": f"75% of values are below {desc[col]['75%']}",
                "90th_percentile": f"90% of values are below {desc[col]['90%']}",
                "mean": f"Average value is {desc[col]['mean']}",
                "std_dev": f"Standard deviation is {desc[col]['std']}",
                "range": f"Values range from {desc[col]['min']} to {desc[col]['max']}",
            }

        # Categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            stats[col] = {
                "unique_values": f"{desc[col]['unique']} distinct values",
                "top_value": f"Most frequent value is '{desc[col]['top']}'",
                "frequency": f"Top value appears {desc[col]['freq']} times",
            }

        return stats

    def _generate_summary(
        self, metadata: Dict, stats: Dict, query: Optional[str]
    ) -> str:
        """Generate human-readable summary"""
        summary = []

        # File info
        summary.append(
            f"Analyzed {metadata['file_type'].upper()} file with "
            f"{metadata['num_rows']} rows and {metadata['num_columns']} columns."
        )

        # Columns overview
        summary.append("\nCOLUMNS:")
        for col, dtype in metadata["dtypes"].items():
            missing = metadata["missing_values"][col]
            summary.append(f"- {col} ({dtype}) - {missing} missing values")

        # Statistics highlights
        summary.append("\nKEY STATISTICS:")
        for col, col_stats in stats.items():
            summary.append(f"\n{col.upper()}:")
            for stat, value in col_stats.items():
                summary.append(f"  • {value}")

        # Query response placeholder
        if query:
            summary.append(
                f"\nNOTE: The system received your query: '{query}'. "
                "For detailed answers, please ask specific questions about the data."
            )

        return "\n".join(summary)

from pathlib import Path
import pandas as pd

# data lives in data/raw relative to the project root (src/ is one level down)
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "tnd_fx_data.xlsx"


def _require_file():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")


def load_basket():
    """USD/TND fixing + interbank rate and the three majors (EUR/USD, GBP/USD, USD/JPY)."""
    _require_file()
    df = pd.read_excel(DATA_FILE, sheet_name="CurrenciesBaskets",
                       parse_dates=["Date"], engine="openpyxl")
    df = df[["Date", "EUR", "GBP", "JPY", "Fixing", "IBrate"]].sort_values("Date")
    return df.set_index("Date")


def load_fixing():
    """BCT official USD/TND fixing with bid/ask quotes."""
    _require_file()
    fx = pd.read_excel(DATA_FILE, sheet_name="Fixing",
                       parse_dates=["Exchange Date"], engine="openpyxl")
    fx = fx.rename(columns={"Exchange Date": "Date", "Mid Price": "Mid"})
    for c in ["Bid", "Ask", "Mid"]:                 # one stray text cell in Bid
        fx[c] = pd.to_numeric(fx[c], errors="coerce")
    fx = fx.dropna(subset=["Mid"])
    return fx.sort_values("Date").set_index("Date")


def load_interbank_panel():
    """Full interbank panel: TND quoted against ~20 currencies (2004 onward)."""
    _require_file()
    ib = pd.read_excel(DATA_FILE, sheet_name="IBrates",
                       parse_dates=["Date"], engine="openpyxl")
    return ib.sort_values("Date").set_index("Date")

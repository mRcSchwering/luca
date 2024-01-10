import pandas as pd
from .util import TABLES_DIR


def write_table(df: pd.DataFrame, name: str, index=False, header=True):
    df = df.copy()
    df.columns = [f"{c}[{t.name}]" for c, t in zip(df.columns, df.dtypes)]
    df.to_csv(TABLES_DIR / name, header=header, index=index)


def read_table(name: str, index_col=False, header=0) -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / name, header=header, index_col=index_col)
    colnames = []
    for col in df.columns:
        colname, dtypestr = col.split("[")
        colnames.append(colname)
        df[col] = df[col].astype(dtypestr[:-1])
    df.columns = colnames
    return df


def to_markdown(df: pd.DataFrame, name: str, descr="", index=False) -> str:
    header = f"**{name}**"
    if len(descr) > 0:
        header += f" {descr}"
    header = "_" + header + "_"
    tab = df.to_markdown(index=index)
    return f"\n{header}\n{tab}\n"

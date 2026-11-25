def print_markdown_table(
    records: list[dict],
    headers: list[str] | None = None,
    bold_header: bool = True,
    bold_first_col: bool = False,
):
    if headers is None:
        headers = list(records[0])

    rows = [[record[h] for h in headers] for record in records]

    def fmt(value, bold=False):
        value = str(value)
        return f"**{value}**" if bold else value

    header = [fmt(h, bold_header) for h in headers]
    print("| " + " | ".join(header) + " |")

    print("| " + " | ".join("---" for _ in headers) + " |")

    for row in rows:
        cells = [fmt(value, bold_first_col and i == 0) for i, value in enumerate(row)]
        print("| " + " | ".join(cells) + " |")

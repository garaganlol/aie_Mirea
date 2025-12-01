from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),

    # 🔥 Новые параметры
    max_hist_columns: int = typer.Option(
        6, help="Максимум числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5, help="Сколько top-значений выводить для категориальных признаков."
    ),
    title: str = typer.Option(
        "EDA-отчёт", help="Заголовок Markdown-отчёта."
    ),
    min_missing_share: float = typer.Option(
        0.2, help="Порог доли пропусков, выше которого колонка считается проблемной."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт.
    """

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество данных
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3. Список колонок, у которых пропуски превышают порог
    problematic_missing = []
    if not missing_df.empty:
        problematic_missing = (
            missing_df[missing_df["missing_share"] >= min_missing_share]
            .index.tolist()
        )

    # 4. Сохраняем артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:

        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Настройки отчёта\n\n")
        f.write(f"- max_hist_columns = **{max_hist_columns}**\n")
        f.write(f"- top_k_categories = **{top_k_categories}**\n")
        f.write(f"- min_missing_share = **{min_missing_share:.2f}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Константные колонки: **{quality_flags['has_constant_columns']}**\n")
        f.write(f"- Высокая кардинальность: **{quality_flags['has_high_cardinality_categoricals']}**\n")
        f.write(f"- Дубликаты ID: **{quality_flags['has_suspicious_id_duplicates']}**\n\n")

        # Проблемные колонки по пропускам
        f.write("## Колонки с пропусками выше порога\n\n")
        if problematic_missing:
            f.write("Порог: **{:.2f}**\n\n".format(min_missing_share))
            for col in problematic_missing:
                f.write(f"- {col}\n")
            f.write("\n")
        else:
            f.write("Нет колонок, превышающих порог.\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Нет категориальных признаков.\n\n")
        else:
            f.write(f"Top-{top_k_categories} значений см. в `top_categories/`.\n\n")

        f.write("## Гистограммы\n\n")
        f.write(f"Максимум колонок: {max_hist_columns}\n\n")

    # 6. Построение графиков
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")


if __name__ == "__main__":
    app()

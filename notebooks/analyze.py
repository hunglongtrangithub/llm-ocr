

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from src import config
    print(config.REPORT_DIR)
    return (config,)


@app.cell
def _(config):
    import json
    import os

    with open(config.REPORT_DIR / "base_results.json", "r") as f:
        base_results = json.load(f)

    with open(config.REPORT_DIR / "rag_results.json", "r") as f:
        rag_results = json.load(f)
    return base_results, rag_results


@app.cell
def _(base_results, rag_results):
    from collections import defaultdict

    def collect_scores(results):
        scores = defaultdict(list)
        for result in results["test_results"]:
            for metric in result["metrics_data"]:
                scores[metric["name"]].append(metric["score"])
        return scores

    rag_scores = collect_scores(rag_results)
    base_scores = collect_scores(base_results)
    rag_scores, base_scores
    return base_scores, rag_scores


@app.cell
def _(base_scores, rag_scores):
    import polars as pl

    rag_df = pl.DataFrame(rag_scores)
    base_df = pl.DataFrame(base_scores)
    rag_df, base_df
    return base_df, rag_df


@app.cell
def _(config, rag_df):
    rag_df.write_csv(config.REPORT_DIR / "rag_scores.csv")
    rag_df.describe()
    return


@app.cell
def _(base_df, config):
    base_df.write_csv(config.REPORT_DIR / "base_scores.csv")
    base_df.describe()
    return


if __name__ == "__main__":
    app.run()

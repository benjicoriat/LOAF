"""Run automated NLP aggregation pipeline.

Produces per-ticker per-period aggregated sentiment features and combined matrices.

Usage: run this script from project root (LOAF) with the venv active.
"""
import os
import glob
import time
import pandas as pd
from tqdm import tqdm

from loaf.config.config import NLPConfig
from loaf.data.nlp_processing import SentimentAnalyzer, WebScraper


LINKS_FOLDER = "./my_project/layer_1/links/"
OUTPUT_FOLDER = "./my_project/layer_1/links_nlp/"
AGG_DIR = os.path.join(OUTPUT_FOLDER, "aggregated")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AGG_DIR, exist_ok=True)


def run_nlp_auto(links_folder=LINKS_FOLDER, output_folder=OUTPUT_FOLDER, agg_dir=AGG_DIR, pause_between_tickers=0.5):
    config = NLPConfig()
    analyzer = SentimentAnalyzer(config)
    scraper = WebScraper(config)

    files = glob.glob(os.path.join(links_folder, "*_links.csv"))
    if not files:
        print(f"No link files found in {links_folder}")
        return

    combined_bar = {}     # period -> {ticker: bar_s}
    combined_p_pos = {}
    combined_p_neg = {}
    combined_p_neu = {}
    combined_N = {}
    combined_sigma = {}

    for path in tqdm(files, desc="Tickers"):
        ticker = os.path.basename(path).replace("_links.csv", "")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        periods = list(df.index.astype(str))

        # Collect URLs and positions
        url_positions = []  # list of (period_index, term)
        urls = []
        for period in df.index:
            for term in df.columns:
                raw = df.at[period, term]
                if isinstance(raw, str) and raw.strip():
                    first_url = raw.split()[0]
                    urls.append(first_url)
                    url_positions.append(period)
                else:
                    # keep placeholder
                    urls.append("")
                    url_positions.append(period)

        # Scrape texts (with caching inside scraper)
        texts = []
        for u in tqdm(urls, desc=f"Scraping {ticker}", leave=False):
            if not u:
                texts.append("")
                continue
            try:
                txt = scraper.scrape_text(u)
                texts.append(txt)
            except Exception:
                texts.append("")

        # Compute sentiment probabilities in batches
        batch_size = config.batch_size
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            probs = analyzer.compute_sentiment_probs(batch)
            scores.extend(probs)

        # Aggregate per period
        rows = []
        # Build mapping period -> list of indices
        period_to_idxs = {}
        for idx, period in enumerate(url_positions):
            period_to_idxs.setdefault(str(period), []).append(idx)

        for period in periods:
            idxs = period_to_idxs.get(str(period), [])
            valid_idxs = [i for i in idxs if i < len(scores) and isinstance(texts[i], str) and texts[i].strip()]

            if not valid_idxs:
                p_pos = 0.5
                p_neg = 0.5
                p_neu = 0.5
                net_mean = 0.0
                N = 0
                sigma = 0.0
            else:
                pos_vals = [scores[i]['pos'] for i in valid_idxs]
                neg_vals = [scores[i]['neg'] for i in valid_idxs]
                neu_vals = [scores[i]['neu'] for i in valid_idxs]
                net_vals = [scores[i]['net'] for i in valid_idxs]

                p_pos = float(sum(pos_vals) / len(pos_vals))
                p_neg = float(sum(neg_vals) / len(neg_vals))
                p_neu = float(sum(neu_vals) / len(neu_vals))
                net_mean = float(sum(net_vals) / len(net_vals))
                N = len(valid_idxs)
                sigma = float(pd.Series(net_vals).std(ddof=0)) if len(net_vals) > 1 else 0.0

            rows.append({
                'period': period,
                'bar_s': net_mean,
                'p_pos': p_pos,
                'p_neg': p_neg,
                'p_neu': p_neu,
                'N': N,
                'sigma': sigma
            })

            # update combined dicts
            combined_bar.setdefault(str(period), {})[ticker] = net_mean
            combined_p_pos.setdefault(str(period), {})[ticker] = p_pos
            combined_p_neg.setdefault(str(period), {})[ticker] = p_neg
            combined_p_neu.setdefault(str(period), {})[ticker] = p_neu
            combined_N.setdefault(str(period), {})[ticker] = N
            combined_sigma.setdefault(str(period), {})[ticker] = sigma

        # Save per-ticker CSV
        out_df = pd.DataFrame(rows).set_index('period')
        out_path = os.path.join(agg_dir, f"{ticker}_sentiment_features.csv")
        out_df.to_csv(out_path)

        # polite pause
        time.sleep(pause_between_tickers)

    # Build combined DataFrames
    # index = sorted union of periods
    all_periods = sorted(set(combined_bar.keys()))
    tickers = [os.path.basename(p).replace("_links.csv", "") for p in files]

    def build_df_from_dict(dict_of_dicts, dtype=float):
        rows = []
        for per in all_periods:
            row = {t: dict_of_dicts.get(per, {}).get(t, None) for t in tickers}
            rows.append(row)
        df = pd.DataFrame(rows, index=all_periods, columns=tickers, dtype=dtype)
        return df

    df_bar = build_df_from_dict(combined_bar, dtype=float)
    df_p_pos = build_df_from_dict(combined_p_pos, dtype=float)
    df_p_neg = build_df_from_dict(combined_p_neg, dtype=float)
    df_p_neu = build_df_from_dict(combined_p_neu, dtype=float)
    df_N = build_df_from_dict(combined_N, dtype=float)
    df_sigma = build_df_from_dict(combined_sigma, dtype=float)

    # Save combined CSVs
    df_bar.to_csv(os.path.join(agg_dir, "aggregated_bar_s.csv"))
    df_p_pos.to_csv(os.path.join(agg_dir, "aggregated_p_pos.csv"))
    df_p_neg.to_csv(os.path.join(agg_dir, "aggregated_p_neg.csv"))
    df_p_neu.to_csv(os.path.join(agg_dir, "aggregated_p_neu.csv"))
    df_N.to_csv(os.path.join(agg_dir, "aggregated_N.csv"))
    df_sigma.to_csv(os.path.join(agg_dir, "aggregated_sigma.csv"))

    print("\n✅ NLP auto aggregation complete. Aggregated files saved to:")
    print(agg_dir)


if __name__ == "__main__":
    run_nlp_auto()

# Sources manifest

Public-domain source texts gathered for the agentsensus research project. All novel
text is unmodified original-language public-domain prose (Three Kingdoms and Dream of
the Red Chamber are Ming/Qing-era classics; the War and Peace translation is the 1904
Maude translation, public domain). The Russia-Ukraine timeline is a from-scratch
chronological compilation of factual events, written in original short-sentence form
(not verbatim Wikipedia prose), sourced from the English Wikipedia timeline article
series.

## 1. three_kingdoms_ch11-60.txt

- **Source**: zh.wikisource.org, 三國演義 (毛宗岗/通行 120-回本), one wikisource page
  per chapter, e.g. `https://zh.wikisource.org/wiki/三國演義/第011回` through
  `.../第060回`. Fetched via `action=raw` (MediaWiki raw wikitext export), simplified
  via OpenCC (t2s) to match the existing ch1-10 reference file.
- **Span**: chapters 11-60 (50 of 50 requested chapters captured, none missing).
- **Size**: 761,831 bytes / 256,990 characters.
- **Format**: matches `three_kingdoms_ch01-10.txt` — each chapter starts with
  `第X回　<title>` (fullwidth space) at the start of a line.

## 2. dream_red_chamber_ch01-80.txt

- **Source**: zh.wikisource.org, 紅樓夢（程乙本） (程偉元/高鶚 120-回通行本), fetched
  from the 8 combined chapter-block pages, e.g.
  `https://zh.wikisource.org/wiki/紅樓夢（程乙本）/第一回　至第十回` through
  `.../第七十一回　至第八十回`. Fetched via `action=raw`, split on `==第X回 ...==`
  section headers, simplified via OpenCC (t2s).
- **Span**: chapters 1-80 (80 of 80 requested chapters captured, none missing).
- **Size**: 1,740,821 bytes / 584,206 characters.
- **Format**: `第X回　<title>` (fullwidth space) at the start of a line, same
  convention as the Three Kingdoms reference file.

## 3. war_and_peace_vol1-3.txt

- **Source**: Project Gutenberg ebook 2600 (Louise & Aylmer Maude translation),
  `https://www.gutenberg.org/files/2600/2600-0.txt`.
- **Span**: Book One through the end of Book Three (Battle of Austerlitz / Prince
  Andrew wounded), i.e. everything before the "BOOK FOUR: 1806" heading. Gutenberg
  license header and footer stripped; original front-matter table of contents and
  "BOOK ONE" / "CHAPTER I" structure preserved as plain lines.
- **Size**: 797,133 bytes / characters (ASCII text).

## 4. russia_ukraine_timeline.txt

- **Source**: English Wikipedia timeline article series (raw wikitext,
  `action=raw`), 14 consecutive articles spanning the full war:
  - Timeline of the 2022 Russian invasion of Ukraine (24 Feb - 7 Apr 2022)
  - Timeline of the Russo-Ukrainian war (8 Apr - 28 Aug 2022)
  - ... (29 Aug - 11 Nov 2022)
  - ... (12 Nov 2022 - 7 Jun 2023)
  - ... (8 Jun 2023 - 31 Aug 2023)
  - ... (1 Sep - 30 Nov 2023)
  - ... (1 Dec 2023 - 31 Mar 2024)
  - ... (1 Apr - 31 Jul 2024)
  - ... (1 Aug - 31 Dec 2024)
  - ... (1 Jan - 31 May 2025)
  - ... (1 Jun - 31 Aug 2025)
  - ... (1 Sep - 31 Dec 2025)
  - ... (1 Jan - 31 May 2026)
  - ... (1 Jun 2026 - present)
  (base URL: `https://en.wikipedia.org/wiki/Timeline_of_the_Russo-Ukrainian_war_(...)`)
- **Method**: each article's wikitext was parsed programmatically by
  `== Month Year ==` / `=== Day Month ===` section headers, wiki markup/citations/
  image captions stripped, and each day's cited text condensed to a short (1-2
  sentence, original-wording, non-verbatim) factual summary. This is a
  ground-truth extraction from the actual cited article text for every date, not an
  LLM free-form summary — an earlier attempt using single large-context
  summarization degraded into generic non-factual filler for some multi-month
  stretches and was discarded in favor of this per-day parse.
- **Span**: 2022-02-24 through 2026-07-19 (today), 1,539 dated lines, one per day
  with events, no duplicate dates, no gaps in coverage detected.
- **Size**: 348,140 bytes / 346,496 characters.
- **Format**: `YYYY-MM-DD: <event>` one line per date, strictly chronological.

## Pre-existing files (not touched by this pass)

- `three_kingdoms_ch01-10.txt` — already present, used as the format reference.
- `three_kingdoms_chibi.txt` — already present, out of scope for this task.

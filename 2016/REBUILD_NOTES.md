# Rebuild notes

Running ledger of bugs we will not reproduce, decisions we are keeping, and places we deliberately left the notebook.

## Scope decided 2026-09-22

- Federal statutes are out.
- First statute crawl is New York and California only. The Maryland criminal-procedure reproduction waits until Maryland is added.
- FindLaw Answers is dead. Training posts are `jonathanli/legal-advice-reddit` (Li, Bhambhoria, Zhu, NLLP 2022, https://aclanthology.org/2022.nllp-1.10). About 98,910 posts, 154 MB, fields `id`, `title`, `body`, `text_label`, `flair_label`, `full_link`, `created_utc`.
- The eleven labels, confirmed from `train.jsonl`: `business`, `contract`, `criminal`, `digital`, `driving`, `employment`, `family`, `housing`, `insurance`, `school`, `wills`. The paper kept the top eleven flairs that are not countries, which is why there is no state label. State remains a query-time input.
- The Hugging Face card has no license field. Cite the paper. Store `full_link`. Do not commit the dump.
- Their train/validation/test split is a few-shot split (test is most of the rows). Pool all three files and make our own stratified split.
- Sampled posts land on the obvious flair (a mass firing is `employment`, landlord fines and a ceiling collapse are `housing`). Some bodies apologize for a guessed flair. Spot-check a sample before treating flair as always the asker's own choice. Do not block the loader on that.
- Timestamps in the sample are around early 2020. This is a snapshot, not a live forum.

## Storage

Posts go in one `POSTS.posts` collection with `section` set to `text_label`, upserted on Reddit `id`. The notebook used one Mongo collection per forum so the collection name could be the label. The algorithm does not care, and dynamic collection names caused the label-cursor bugs below.

Laws stay one collection per state (`NY`, `CA`). Articles stay one collection, with the Nolo area on the document.

Matrix row `i` and display record `i` are written in the same training job. Query time indexes that table. It does not run the routing query again.

## Bugs, do not reproduce

Each of these gets a test when the code that replaces it is written.

- Law paths call `' '.join` on `getTokens`, which already returns a string, so tokens become single characters. That is the likely cause of the saved `ValueError: empty vocabulary` (notebook law-vectorizer cell).
- Labels come from a second unsorted `find()`, and from whichever database `db` last pointed at.
- The article vectorizer iterates the mapping dict's keys instead of the Nolo sections mapped to each forum.
- The filename sanitizer assigns from the original string on every character, so only `:` is removed.
- No guard for an empty section.
- Routing files are loaded with `eval`.
- Tokenization walks every database on the Mongo server.
- A new `PorterStemmer` and stopword list are built per word and per document.

Tests cover all eight: `test_train.py`, `test_validate.py`, `test_text.py`. An empty law route returns None instead of raising `empty vocabulary`. Routing files are read with `json.loads`.

## Design decisions kept

- Strip all digits.
- Porter stemming, NLTK English stopwords, bag-of-words TF-IDF, `max_features=10000`, `lowercase=False`.
- `MultinomialNB`, one label per question.
- Top 5 by cosine similarity, only inside the routed area.
- Article titles that name another state are dropped. The user's state is preferred, then titles that name no state.

## Routing, decided 2026-09-22

- Hand-map the eleven flairs to Nolo areas and to statute path prefixes. No LLM generator.
- A route is a path prefix: code, then title (the second hierarchy level the Justia spike actually finds). Match documents whose stored `section` list starts with that prefix. Do not match a bare title string.

## Posts loaded 2026-09-22

`POSTS.posts` has 98,910 documents, the full pooled dataset, revision `f105b9d763743e20d2f3b8e33f73055ad414e7c5`. No rows were skipped. Six posts have an empty body. Median text length is 988 characters. Counts: housing 23,874, employment 15,926, family 11,938, driving 9,423, criminal 8,684, business 5,763, digital 5,270, contract 5,170, school 4,323, insurance 4,316, wills 4,223.

No per-class cap. The smallest class still has 4,223 posts, and a cap at 1,500 would throw away the class prior `MultinomialNB` is supposed to learn.

## Token oracle, NLTK 3.10.3

The notebook prints this token list for the first sentence of the CPL § 450.30 example:

`appeal defend sentenc author subdivis two section may base upon ground sentenc either invalid matter law harsh excess`

`get_tokens` on that sentence, with NLTK 3.10.3 Porter and the current English stopword list, matches that list except it also emits `b`. Punctuation stripping turns `(b)` into `b`, and `b` is not a stopword. `(a)` disappears because `a` is a stopword. `upon` is still kept, so this is not a stopword-list change. The printed list is the oracle. The extra `b` is the recorded diff. Single letters are not dropped to make the lists match.

## Post classifier, 2026-09-22

Held-out 20% stratified split, `random_state=2016`, vectorizer fit on the train split only. Accuracy 0.7973, macro-F1 0.7125, 19,782 test posts. The serving model is a refit on all 98,910 posts. Vectorizer is `max_features=10000`, `lowercase=False`, `norm=l2`, token pattern `(?u)\b\w\w+\b`. `MultinomialNB` defaults. scikit-learn 1.9.1, NLTK 3.10.3. Artifact: `2016/artifacts/posts_classifier.joblib`. Metrics: `2016/artifacts/posts_metrics.json`.

Recall is high on the large flairs (housing 0.973, family 0.949, employment 0.948, driving 0.921) and low where the class is small and the language overlaps (contract 0.290, insurance 0.366, business 0.406). That is the class prior doing what it was left in place to do.

Token fields were written on `POSTS.posts` (98,910), `ARTICLES.articles` (6,172, mid-crawl, so the next train rewrites them), and `LAWS.CA` (161,426). `LAWS.NY` was tokenized later, after the Senate load, 10,816 sections.

## Law rankers, 2026-09-22

One TF-IDF matrix per flair per state, fit only on sections whose path starts with a mapped prefix. No route was empty. The notebook law vectorizer is `lowercase=False` with no `max_features` cap; the token pattern is pinned to `(?u)\b\w\w+\b`. Each artifact holds the vectorizer, the sparse matrix, and the display rows in that same order: `2016/artifacts/laws/{NY,CA}/{flair}.joblib`. Counts are in `2016/artifacts/laws_report.json`. New York employment is 41 sections (Articles 6 and 19). California employment is 954 (Labor Code division 2). California school is the largest, 11,520, because the route is the whole Education Code.

## Article rankers, 2026-09-22

994 articles were missing tokens after the crawl finished. Those were tokenized, then one matrix was fit per flair on the mapped Nolo areas only. Digital and school were skipped because their article lists are empty. Contract is 744 articles in Business Formation and Small Claims, and the rows include "Contracts 101: Make a Legally Valid Contract". Display rows are title, url, and area. No article body is stored in the artifact. Files are `2016/artifacts/articles/{flair}.joblib`. Counts are in `2016/artifacts/articles_report.json`.

## Overtime question, 2026-09-22

Question: "i work more than 40 hours a week but my boss wont pay me for overtime". `get_tokens` yields `work hour week boss wont pay overtim`. The digits in 40 are gone.

Both states classify as `employment` with classifier score 0.945. That score is not a calibrated probability.

New York laws, cosine: LAB § 661 records of employers (0.317), § 195 notice and record-keeping (0.204), § 196-E construction reporting pay (0.146), § 652 minimum wage (0.122), § 190 definitions (0.097). The route is only Articles 6 and 19, payment of wages and minimum wage, so the hours statutes are not in the matrix. Top article: "What's Your Unpaid Wage Claim Worth in New York?"

California laws: LAB § 556 (0.415), § 1815 (0.376), § 751.8 (0.362), § 513 (0.361), § 1454 (0.340). Top article: "California Wage and Hour Laws". The state filter kept California titles and dropped other states.

1,206 posts mention the word "flair." In a sample of 20, the asker is choosing a flair and often says they may have it wrong ("hopefully this is the correct flair," "please fix my flair"). That is the asker's own label, with ordinary self-doubt. It is not evidence that moderators replaced the label. The label stays the training target.

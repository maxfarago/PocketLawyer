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

1,206 posts mention the word "flair." In a sample of 20, the asker is choosing a flair and often says they may have it wrong ("hopefully this is the correct flair," "please fix my flair"). That is the asker's own label, with ordinary self-doubt. It is not evidence that moderators replaced the label. The label stays the training target.

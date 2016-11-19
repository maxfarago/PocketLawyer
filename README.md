# Pocket Lawyer: a real-time legal advice web app
#### Powered by Python and MongoDB; deployed with Flask and jQuery

This is the repo for Pocket Lawyer, a web app that takes a user's legal question and provides links to relevant articles and laws.  Pocket Lawyer uses geolocation to determine a user's home state and give jurisdiction-specific legal advice and statutes from that state's legal code.

Pocket Lawyer is available to use online [here]("http://193.91.67.209/pocketLawyer/").  The source code for the basic algorithms working under the hood can be seen in the 'Pocket Lawyer Basics' Jupyter Notebook.

#### Algorithm

Pocket Lawyer works by categorizing a user query using a multi-class classifier trained on posts from law-related message boards.  The boards are divided into forums for different areas of law, and a poster asks their question in the forum governing their issue.  All the posts are scraped and used to train a model with the forum areas as labels.  The model essentially predicts which forum the Pocket Lawyer user would have posted their question.

The area of law predicted by the classifier tells Pocket Lawyer which articles and laws to search through for relevant results.  For example: if a user's query is determined to be a **Family Law** question, Pocket Lawyer will search through the articles that give advice about family law related issues.  After locating the user's state, Pocket Lawyer also searches through the portions of that state's legal code that govern family and domestic legal issues.  The most relevant articles and laws are returned to the user.

Pocket Lawyer uses a unique combination of vectorization, classification, and cosine similarity to retrieve accurate results.

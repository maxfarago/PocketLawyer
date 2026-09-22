"""Scope of this rebuild. Change it here, not in scattered call sites."""

# Federal statutes are out of scope. First crawl is two states.
STATES = ("NY", "CA")

POSTS_DB = "POSTS"
ARTICLES_DB = "ARTICLES"
LAWS_DB = "LAWS"

# One collection. The forum/flair label is a field, not a Mongo collection name.
POSTS_COLLECTION = "posts"
ARTICLES_COLLECTION = "articles"

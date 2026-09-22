from pl2016.config import STATES
from pl2016.posts.reddit import DATASET_ID, FIELDS, SPLITS


def test_jurisdictions_are_ny_and_ca_only():
    assert STATES == ("NY", "CA")


def test_posts_come_from_the_li_2022_reddit_dataset():
    assert DATASET_ID == "jonathanli/legal-advice-reddit"
    assert SPLITS == ("train", "validation", "test")
    assert "text_label" in FIELDS
    assert "body" in FIELDS
    assert "full_link" in FIELDS

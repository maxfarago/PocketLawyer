from pl2016.scrape.nolo import encyclopedia_urls, parse_article

SITEMAP = """
<urlset>
  <loc>https://www.nolo.com/legal-encyclopedia/question-boss-underpaying-me-28044.html</loc>
  <loc>https://www.nolo.com/legal-encyclopedia/wills/all</loc>
  <loc>https://www.nolo.com/front-page</loc>
  <loc>https://www.nolo.com/legal-encyclopedia/question-boss-underpaying-me-28044.html</loc>
</urlset>
"""

ARTICLE = """
<script type="application/ld+json">
{"@graph":[
  {"@type":"Article","headline":"What If My Employer Pays Me Less Than Minimum Wage?","articleBody":"The boss did not pay."},
  {"@type":"BreadcrumbList","itemListElement":[
    {"position":1,"item":{"name":"Learn by Legal Issue","@id":"https://www.nolo.com/legal-encyclopedia"}},
    {"position":2,"item":{"name":"Employment Law","@id":"https://www.nolo.com/legal-encyclopedia/hr-employment-law"}},
    {"position":3,"item":{"name":"Employee Rights","@id":"https://www.nolo.com/legal-encyclopedia/employee-rights"}}
  ]}
]}
</script>
"""


def test_sitemap_keeps_encyclopedia_articles_only():
    urls = encyclopedia_urls(SITEMAP)
    assert urls == [
        "https://www.nolo.com/legal-encyclopedia/question-boss-underpaying-me-28044.html"
    ]


def test_parse_article_uses_the_second_breadcrumb_as_the_area():
    doc = parse_article(ARTICLE, "https://www.nolo.com/legal-encyclopedia/question-boss-underpaying-me-28044.html")
    assert doc is not None
    assert doc["title"].startswith("What If My Employer")
    assert doc["text"] == "The boss did not pay."
    assert doc["area"] == "Employment Law"
    assert doc["area_url"].endswith("/hr-employment-law")


def test_parse_article_skips_a_page_with_no_article_node():
    assert parse_article("<html><title>Wills</title></html>", "https://www.nolo.com/legal-encyclopedia/wills") is None

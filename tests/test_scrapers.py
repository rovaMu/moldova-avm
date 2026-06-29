"""Tests that each site scraper parses representative (and dirty) HTML."""

from app.scraper import parser as p


NINE99_HTML = """
<html><body>
<ul>
  <li class="ads-list-photos-item" data-listing>
    <a class="ads-list-photo-item-title" href="/ru/123">Apartament Bloc nou Centru</a>
    <span class="ads-list-photo-item-price">85 000 €</span>
    <ul>
      <li class="is-wrapped"><span class="m-label">Suprafața</span><span class="m-value">65 m²</span></li>
      <li class="is-wrapped"><span class="m-label">Etaj</span><span class="m-value">3/9</span></li>
      <li class="is-wrapped"><span class="m-label">Sector</span><span class="m-value">Centru</span></li>
      <li class="is-wrapped"><span class="m-label">Reparație</span><span class="m-value">Euroreparație</span></li>
    </ul>
  </li>
  <li class="ads-list-photos-item">
    <a class="ad-title" href="">Garbage row with no price</a>
  </li>
</ul>
</body></html>
"""


def test_999_apartment_parse():
    scraper = p.NineNineNineScraper()
    listings = scraper.parse_normalized(NINE99_HTML, p.APARTMENT)
    assert len(listings) == 2
    first = listings[0]
    assert first.source == "999.md"
    assert first.price_eur == 85000.0
    assert first.sqm == 65.0
    assert first.current_floor == 3
    assert first.total_floors == 9
    assert first.is_middle_floor == 1
    assert first.sector == p.SECTOR_CENTRU
    assert first.is_new_building == 1
    assert first.renovation_level == 3
    # The junk row must still normalise without raising.
    assert listings[1].price_eur is None


MAKLER_HTML = """
<html><body>
<div class="list-item">
  <h3 class="title">Apartament Hrușciovka Botanica</h3>
  <span class="price">42000 EUR</span>
  <ul class="characteristics">
    <li>Suprafața: 45 m²</li>
    <li>Etaj: 2 din 5</li>
    <li>Serie: 143</li>
    <li>Reparație: Fără reparație</li>
  </ul>
</div>
</body></html>
"""


def test_makler_old_series_tags():
    scraper = p.MaklerScraper()
    listings = scraper.parse_normalized(MAKLER_HTML, p.APARTMENT)
    assert len(listings) == 1
    item = listings[0]
    assert item.price_eur == 42000.0
    assert item.sqm == 45.0
    assert item.sector == p.SECTOR_BOTANICA
    # Hrușciovka / Serie 143 => old building
    assert item.is_new_building == 0
    assert item.renovation_level == 1


PROIMOBIL_HTML = """
<html><head>
<script type="application/json">
{"listings":[
  {"title":"Complex Bloc nou","price":"120000","area":"90 m²",
   "sector":"Telecentru","construction_phase":"Variantă Albă",
   "finish":"Variantă albă","floor":"5/10"}
]}
</script>
</head><body></body></html>
"""


def test_proimobil_json_script():
    scraper = p.ProimobilScraper()
    listings = scraper.parse_normalized(PROIMOBIL_HTML, p.APARTMENT)
    assert len(listings) == 1
    item = listings[0]
    assert item.source == "proimobil.md"
    assert item.price_eur == 120000.0
    assert item.sqm == 90.0
    assert item.sector == p.SECTOR_TELECENTRU
    assert item.is_new_building == 1
    assert item.renovation_level == 1  # variantă albă -> raw shell


def test_oximobil_dom_fallback():
    html = """
    <html><body>
      <article class="property-card">
        <h2 class="title">Casă Vilă Durlești</h2>
        <span class="price">€250000</span>
        <ul>
          <li>Suprafața: 200 m²</li>
          <li>Teren: 6 ari</li>
          <li>Niveluri: 2 niveluri</li>
          <li>Finisaj: Lux</li>
        </ul>
      </article>
    </body></html>
    """
    scraper = p.OximobilScraper()
    listings = scraper.parse_normalized(html, p.HOUSE)
    assert len(listings) == 1
    item = listings[0]
    assert item.price_eur == 250000.0
    assert item.sqm == 200.0
    assert item.land_ari == 6.0
    assert item.levels == 2
    assert item.house_sub_type == p.HOUSE_VILLA
    assert item.sector == p.SECTOR_SUBURBS
    assert item.renovation_level == 4


LARA_HTML = """
<html><body>
<table>
  <tr><th>Sector</th><th>Suprafata</th><th>Pret</th><th>Etaj</th><th>Tip</th></tr>
  <tr><td>Râșcani</td><td>50 m²</td><td>55000</td><td>4/9</td><td>Bloc vechi</td></tr>
  <tr><td>Ciocana</td><td>70 mp</td><td>72000</td><td>1/5</td><td>Bloc nou</td></tr>
</table>
</body></html>
"""


def test_lara_table_parse():
    scraper = p.LaraScraper()
    listings = scraper.parse_normalized(LARA_HTML, p.APARTMENT)
    assert len(listings) == 2
    assert listings[0].sector == p.SECTOR_RASCANI
    assert listings[0].is_new_building == 0
    assert listings[1].sector == p.SECTOR_CIOCANA
    assert listings[1].is_middle_floor == 0  # floor 1 penalty


def test_parse_empty_and_malformed_html():
    for scraper in p.SCRAPERS.values():
        assert scraper.parse("") == []
        assert scraper.parse("<html><body><div>nope</div></body></html>") == []
        # Must not raise on broken markup.
        scraper.parse("<html><body><table><tr><td>")

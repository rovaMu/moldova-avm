"""Target site scrapers + multilingual regex data cleaners / normalizers.

This module is deliberately split into two layers:

1. **Normalizers** - pure, side-effect free functions that map messy
   Romanian/Russian listing strings onto the numeric feature space used by
   the regression engine. They are heavily defensive: every function accepts
   arbitrary dirty input (``None``, empty strings, mixed encodings, junk) and
   always returns a well typed value, falling back to the documented
   fail-safe defaults instead of raising.

2. **Scrapers** - one small ``BeautifulSoup`` based parser per target
   platform (999.md, Makler.md, Proimobil.md, Oximobil.md, Lara.md). Each
   exposes ``parse(html) -> list[RawListing]`` returning raw string fields,
   plus ``parse_normalized`` which runs the raw output through the normalizer
   pipeline to produce model-ready :class:`NormalizedListing` records.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Property type discriminator values
# ---------------------------------------------------------------------------
APARTMENT = "apartment"
HOUSE = "house"


# ---------------------------------------------------------------------------
# Low level text helpers
# ---------------------------------------------------------------------------
# Explicit Romanian diacritic folding. Done by table (not NFKD) so that
# Cyrillic characters such as "й" (which NFKD would decompose into и + a
# combining breve) are left intact and keep matching Russian keywords.
_ROMANIAN_FOLD = str.maketrans(
    {
        "ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t",
        "ş": "s", "ţ": "t", "Ă": "a", "Â": "a", "Î": "i",
        "Ș": "s", "Ț": "t", "Ş": "s", "Ţ": "t",
    }
)


def _norm_text(value: Any) -> str:
    """Return a lowercase, accent-folded, whitespace-collapsed string.

    Romanian diacritics (ă, â, î, ș, ț and their cedilla variants) are folded
    to ASCII so that "Râșcani", "Rascani" and "Rîşcani" all compare equal.
    Cyrillic is preserved (only lowercased) so Russian keywords still match.
    """
    if value is None:
        return ""
    text = str(value).translate(_ROMANIAN_FOLD)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_float(value: Any) -> Optional[float]:
    """Best-effort numeric extraction from arbitrary input."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("\u00a0", " ")
    # Capture a number that may carry grouped thousands and/or a decimal part.
    match = re.search(r"-?\d[\d .,]*\d|-?\d", text)
    if not match:
        return None
    token = match.group(0).replace(" ", "")  # spaces are always thousands sep
    if "," in token and "." in token:
        # Both present: the right-most separator is the decimal point.
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = _resolve_single_separator(token, ",")
    elif "." in token:
        token = _resolve_single_separator(token, ".")
    try:
        return float(token)
    except ValueError:
        return None


def _resolve_single_separator(token: str, sep: str) -> str:
    """Disambiguate a single separator char as thousands vs. decimal.

    Multiple occurrences => thousands grouping (remove them). A single
    occurrence followed by exactly three digits => thousands; otherwise it is
    treated as a decimal point.
    """
    parts = token.split(sep)
    if len(parts) > 2:
        return token.replace(sep, "")
    if len(parts[-1]) == 3:
        return token.replace(sep, "")
    return token.replace(sep, ".")


# ---------------------------------------------------------------------------
# Normalization maps
# ---------------------------------------------------------------------------
SECTOR_CENTRU = 0
SECTOR_RASCANI = 1
SECTOR_BOTANICA = 2
SECTOR_BUIUCANI = 3
SECTOR_CIOCANA = 4
SECTOR_TELECENTRU = 5
SECTOR_SUBURBS = 6
SECTOR_DEFAULT = SECTOR_BOTANICA  # mid-market average fail-safe

# Order matters: more specific / suburb tokens are checked too, but we scan
# all and pick the first hit by priority list below.
# NOTE: Telecentru must precede Centru because "telecentru"/"телецентр"
# both contain the substring "centru"/"центр".
_SECTOR_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    (SECTOR_TELECENTRU, ("telecentru", "телецентр")),
    (SECTOR_CENTRU, ("centru", "центр")),
    (SECTOR_RASCANI, ("rascani", "ryshkanovka", "рышкановка", "рашкановка")),
    (SECTOR_BOTANICA, ("botanica", "ботаника")),
    (SECTOR_BUIUCANI, ("buiucani", "буюканы", "буюкань")),
    (SECTOR_CIOCANA, ("ciocana", "чеканы", "чокана")),
    (
        SECTOR_SUBURBS,
        (
            "durlesti",
            "cricova",
            "bardar",
            "stauceni",
            "codru",
            "bacioi",
            "kodru",
            "крикова",
            "ставчены",
            "дурлешты",
        ),
    ),
]

# Human readable sector labels used in the public formula payload.
SECTOR_LABELS = {
    SECTOR_CENTRU: "Centru",
    SECTOR_RASCANI: "Rascani",
    SECTOR_BOTANICA: "Botanica",
    SECTOR_BUIUCANI: "Buiucani",
    SECTOR_CIOCANA: "Ciocana",
    SECTOR_TELECENTRU: "Telecentru",
    SECTOR_SUBURBS: "Suburbs",
}


def normalize_sector(value: Any) -> int:
    """Map a Chișinău location string to its numeric sector code."""
    text = _norm_text(value)
    if not text:
        return SECTOR_DEFAULT
    for code, keywords in _SECTOR_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return code
    return SECTOR_DEFAULT


_NEW_BUILDING_KEYWORDS = (
    "bloc nou",
    "blocuri noi",
    "varianta alba",  # accent-folded form of "Varianta albă"
    "dat in exploatare",
    "новострой",
    "сдан в эксплуатацию",
    "новостройка",
)
_OLD_BUILDING_KEYWORDS = (
    "bloc vechi",
    "hrusciovka",  # accent-folded "Hrușciovka"
    "хрущевка",
    "хрущёвка",
    "seria 143",
    "serie 143",
    "moldoveneasca",
    "вторичный рынок",
    "брежневка",
    "breznevka",
)


def normalize_building_era(value: Any) -> int:
    """Return 1 for modern/new buildings, 0 for Soviet/old era.

    New-building keywords take precedence over old ones when both appear.
    Fail-safe default is 1 (modern).
    """
    text = _norm_text(value)
    if not text:
        return 1
    for kw in _NEW_BUILDING_KEYWORDS:
        if kw in text:
            return 1
    for kw in _OLD_BUILDING_KEYWORDS:
        if kw in text:
            return 0
    # "MS" Soviet classification tag - match as a standalone token only.
    if re.search(r"(?<![a-z])ms(?![a-z])", text):
        return 0
    return 1


# Ordinal repair scale 1..4. Higher tier keywords are checked first so that
# e.g. "euro lux" resolves to 4 rather than 3.
_REPAIR_RULES: list[tuple[int, tuple[str, ...]]] = [
    (4, ("lux", "design individual", "эксклюзив", "премиум", "premium")),
    (3, ("euroreparatie", "euro", "евроремонт")),
    (
        2,
        (
            "reparatie cosmetica",
            "cosmeticeschi",
            "cosmetic",
            "косметический ремонт",
            "косметический",
            "жилое состояние",
        ),
    ),
    (
        1,
        (
            "fara reparatie",
            "varianta alba",
            "без ремонта",
            "черновая отделка",
            "черновая",
        ),
    ),
]


def normalize_repair_level(value: Any) -> int:
    """Map renovation descriptions onto the ordinal scale 1..4 (default 3)."""
    text = _norm_text(value)
    if not text:
        return 3
    for level, keywords in _REPAIR_RULES:
        for kw in keywords:
            if kw in text:
                return level
    return 3


_SQM_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:m\s*²|m2|m\^?2|mp|кв\.?\s*м|кв\.?\s*метр)",
    re.IGNORECASE,
)


def extract_sqm(value: Any) -> Optional[float]:
    """Extract living/total area in square metres as a clean float."""
    if value is None:
        return None
    text = str(value).replace("\u00a0", " ")
    match = _SQM_PATTERN.search(text)
    if match:
        return _to_float(match.group(1))
    # Fall back to a bare number when the field is clearly an area field.
    return _to_float(text)


# Land area normalisation. Target unit is "ari" (== "sote"/"соток").
# 1 ari == 100 m²; 1 ha == 100 ari; 1 sote == 1 ari.
_LAND_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*ha\b", re.IGNORECASE), 100.0),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:гектар|га)\b", re.IGNORECASE), 100.0),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*ari\b", re.IGNORECASE), 1.0),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:sote|sotka)\b", re.IGNORECASE), 1.0),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*сот", re.IGNORECASE), 1.0),
]


def extract_land_ari(value: Any) -> Optional[float]:
    """Parse a land footprint string and normalise it to Ari (sote)."""
    if value is None:
        return None
    text = str(value).replace("\u00a0", " ")
    for pattern, factor in _LAND_PATTERNS:
        match = pattern.search(text)
        if match:
            number = _to_float(match.group(1))
            if number is not None:
                return round(number * factor, 4)
    return None


_FLOOR_SPLIT_PATTERNS = [
    re.compile(r"(\d+)\s*/\s*(\d+)"),                       # "3/9"
    re.compile(r"(\d+)\s*de\s*la\s*(\d+)", re.IGNORECASE),  # "5 de la 10"
    re.compile(r"(\d+)\s*din\s*(\d+)", re.IGNORECASE),      # "5 din 10"
    re.compile(r"(\d+)\s*из\s*(\d+)", re.IGNORECASE),       # "5 из 10"
    re.compile(r"(\d+)\s*эт.*?(\d+)\s*эт", re.IGNORECASE),  # "5 эт из 10 эт"
]


def split_floor(value: Any) -> tuple[Optional[int], Optional[int]]:
    """Split a complex floor field into (current_floor, total_floors)."""
    if value is None:
        return None, None
    text = str(value)
    for pattern in _FLOOR_SPLIT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                current = int(match.group(1))
                total = int(match.group(2))
                return current, total
            except (ValueError, IndexError):
                continue
    # Single number fallback => current floor only.
    single = re.search(r"\d+", text)
    if single:
        return int(single.group(0)), None
    return None, None


def compute_is_middle_floor(
    current_floor: Optional[int], total_floors: Optional[int]
) -> int:
    """1 for desirable middle floors, 0 for ground/top floor penalty.

    When data is missing we default to 1 (no penalty) to avoid biasing the
    model toward an unverified penalty.
    """
    if current_floor is None or total_floors is None or total_floors <= 0:
        return 1
    if current_floor <= 1 or current_floor >= total_floors:
        return 0
    return 1


HOUSE_INDIVIDUAL = 0
HOUSE_TOWNHOUSE = 1
HOUSE_VILLA = 2

_HOUSE_SUBTYPE_RULES: list[tuple[int, tuple[str, ...]]] = [
    (HOUSE_VILLA, ("vila", "casa de vacanta", "вилла", "дача", "коттедж")),
    (HOUSE_TOWNHOUSE, ("townhouse", "town house", "duplex", "таунхаус")),
    (HOUSE_INDIVIDUAL, ("casa individuala", "individual", "particular")),
]


def normalize_house_sub_type(value: Any) -> int:
    """Map a house description onto its structural sub-type code (default 0)."""
    text = _norm_text(value)
    if not text:
        return HOUSE_INDIVIDUAL
    for code, keywords in _HOUSE_SUBTYPE_RULES:
        for kw in keywords:
            if kw in text:
                return code
    return HOUSE_INDIVIDUAL


_LEVELS_PATTERN = re.compile(
    r"(\d+)\s*(?:nivel|niveluri|etaj|этаж|уровн)", re.IGNORECASE
)


def extract_levels(value: Any) -> Optional[int]:
    """Extract the number of structural levels/floors of a house."""
    if value is None:
        return None
    text = str(value)
    match = _LEVELS_PATTERN.search(text)
    if match:
        return int(match.group(1))
    number = _to_float(text)
    return int(number) if number is not None else None


def extract_price_eur(value: Any) -> Optional[float]:
    """Extract a EUR price as float. Returns None on unparseable junk."""
    return _to_float(value)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
@dataclass
class RawListing:
    """Loosely-typed listing fields exactly as scraped from a page."""

    source: str
    property_type: str
    title: str = ""
    price_raw: str = ""
    area_raw: str = ""
    land_raw: str = ""
    floor_raw: str = ""
    sector_raw: str = ""
    building_raw: str = ""
    repair_raw: str = ""
    house_type_raw: str = ""
    levels_raw: str = ""
    listed_at: Optional[str] = None
    url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedListing:
    """Model-ready numeric feature record."""

    source: str
    property_type: str
    price_eur: Optional[float]
    sqm: Optional[float]
    sector: int
    is_new_building: int
    renovation_level: int
    # Apartment-specific
    current_floor: Optional[int] = None
    total_floors: Optional[int] = None
    is_middle_floor: int = 1
    # House-specific
    land_ari: Optional[float] = None
    house_sub_type: int = 0
    levels: Optional[int] = None
    # Provenance
    listed_at: Optional[str] = None
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_listing(raw: RawListing) -> NormalizedListing:
    """Convert a :class:`RawListing` into numeric features defensively."""
    # Aggregate free-text fields so keyword detection is resilient to the
    # exact field a site stuffed the information into.
    blob = " ".join(
        str(x)
        for x in (
            raw.title,
            raw.building_raw,
            raw.repair_raw,
            raw.house_type_raw,
            raw.sector_raw,
        )
        if x
    )

    price = extract_price_eur(raw.price_raw)
    sqm = extract_sqm(raw.area_raw)
    sector = normalize_sector(raw.sector_raw or blob)
    is_new = normalize_building_era(raw.building_raw or blob)
    repair = normalize_repair_level(raw.repair_raw or blob)

    current_floor, total_floors = split_floor(raw.floor_raw)
    is_middle = compute_is_middle_floor(current_floor, total_floors)

    land_ari = extract_land_ari(raw.land_raw)
    house_sub_type = normalize_house_sub_type(raw.house_type_raw or blob)
    levels = extract_levels(raw.levels_raw)

    return NormalizedListing(
        source=raw.source,
        property_type=raw.property_type,
        price_eur=price,
        sqm=sqm,
        sector=sector,
        is_new_building=is_new,
        renovation_level=repair,
        current_floor=current_floor,
        total_floors=total_floors,
        is_middle_floor=is_middle,
        land_ari=land_ari,
        house_sub_type=house_sub_type,
        levels=levels,
        listed_at=raw.listed_at or datetime.now(timezone.utc).isoformat(),
        url=raw.url,
    )


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------
class BaseScraper:
    """Common scraper behaviour. Subclasses implement :meth:`parse`."""

    source: str = "base"
    apartment_paths: tuple[str, ...] = ()
    house_paths: tuple[str, ...] = ()

    @staticmethod
    def _text(node: Any) -> str:
        if node is None:
            return ""
        if isinstance(node, str):
            return node.strip()
        return node.get_text(" ", strip=True)

    def parse(self, html: str, property_type: str = APARTMENT) -> list[RawListing]:
        raise NotImplementedError

    def parse_normalized(
        self, html: str, property_type: str = APARTMENT
    ) -> list[NormalizedListing]:
        return [normalize_listing(raw) for raw in self.parse(html, property_type)]


class NineNineNineScraper(BaseScraper):
    """999.md - high volume classifieds board.

    Listings render as ``ad-link`` style cards; structured attributes live in
    a feature grid whose cells carry classes such as ``is-wrapped`` and
    ``m-value``. Floor fields like "3/9" or "5 de la 10" are split here.
    """

    source = "999.md"
    apartment_paths = ("/real-estate/apartments-and-rooms",)
    house_paths = ("/real-estate/houses",)

    def parse(self, html: str, property_type: str = APARTMENT) -> list[RawListing]:
        soup = BeautifulSoup(html or "", "html.parser")
        listings: list[RawListing] = []

        cards = soup.select(
            "li.ads-list-photos-item, div.ads-list-photo-item, "
            "div.ad-card, li.ad-item, [data-listing]"
        )
        for card in cards:
            title = self._text(
                card.select_one(".ads-list-photo-item-title, .ad-title, a")
            )
            price = self._text(
                card.select_one(".ads-list-photo-item-price, .ad-price, .price")
            )

            # Build a feature dict from the feature grid cells.
            features: dict[str, str] = {}
            for cell in card.select(".is-wrapped, .m-value, .feature, li"):
                label_node = cell.select_one(".m-label, .label, dt, .feature-name")
                value_node = cell.select_one(".m-value, .value, dd, .feature-value")
                label = _norm_text(self._text(label_node))
                value = self._text(value_node) or self._text(cell)
                if label:
                    features[label] = value

            listings.append(
                RawListing(
                    source=self.source,
                    property_type=property_type,
                    title=title,
                    price_raw=price,
                    area_raw=_first_feature(
                        features, ("suprafata", "suprafata totala", "площадь")
                    ),
                    land_raw=_first_feature(
                        features, ("teren", "suprafata teren", "участок")
                    ),
                    floor_raw=_first_feature(features, ("etaj", "этаж")),
                    sector_raw=_first_feature(
                        features, ("sector", "regiune", "район", "localitate")
                    )
                    or title,
                    building_raw=_first_feature(
                        features, ("tip", "tipul casei", "тип", "категория")
                    )
                    or title,
                    repair_raw=_first_feature(
                        features, ("reparatie", "stare", "ремонт", "состояние")
                    ),
                    house_type_raw=_first_feature(
                        features, ("tip imobil", "tip casa", "тип дома")
                    )
                    or title,
                    levels_raw=_first_feature(
                        features, ("nivel", "niveluri", "этажность")
                    ),
                    url=_first_href(card),
                )
            )
        return listings


class MaklerScraper(BaseScraper):
    """Makler.md - historical secondary market.

    Listings render as list grids; old Soviet classification tags such as
    "Serie 143", "MS", "Hrușciovka" and "Breznevka" appear in the title or a
    series/type cell and are preserved into ``building_raw`` for era mapping.
    """

    source = "makler.md"
    apartment_paths = ("/real-estate/real-estate-for-sale",)
    house_paths = ("/real-estate/real-estate-for-sale",)

    def parse(self, html: str, property_type: str = APARTMENT) -> list[RawListing]:
        soup = BeautifulSoup(html or "", "html.parser")
        listings: list[RawListing] = []

        rows = soup.select(
            "div.list-item, li.list-item, div.ad, tr.listing, .catalog-item"
        )
        for row in rows:
            title = self._text(row.select_one(".title, h3, h2, a"))
            price = self._text(row.select_one(".price, .cost, .list-price"))

            features: dict[str, str] = {}
            for cell in row.select(".param, .params li, .characteristics li, li"):
                text = self._text(cell)
                if ":" in text:
                    label, _, value = text.partition(":")
                    features[_norm_text(label)] = value.strip()

            series = _first_feature(features, ("serie", "seria", "серия", "тип"))
            building_blob = " ".join(filter(None, [title, series]))

            listings.append(
                RawListing(
                    source=self.source,
                    property_type=property_type,
                    title=title,
                    price_raw=price,
                    area_raw=_first_feature(
                        features, ("suprafata", "площадь", "s общая")
                    ),
                    land_raw=_first_feature(features, ("teren", "участок", "сот")),
                    floor_raw=_first_feature(features, ("etaj", "этаж")),
                    sector_raw=_first_feature(
                        features, ("sector", "raion", "район", "город")
                    )
                    or title,
                    building_raw=building_blob,
                    repair_raw=_first_feature(
                        features, ("reparatie", "ремонт", "состояние")
                    ),
                    house_type_raw=_first_feature(features, ("tip", "тип дома"))
                    or title,
                    levels_raw=_first_feature(features, ("nivel", "этажность")),
                    url=_first_href(row),
                )
            )
        return listings


class _ResidentialComplexScraper(BaseScraper):
    """Shared logic for Proimobil.md / Oximobil.md premium agency listings.

    These sites embed structured metadata in ``<script>`` tags (often
    JSON-LD or a custom ``window.__DATA__`` payload). We prefer that JSON when
    available and fall back to DOM cards otherwise. Construction-phase phrases
    ("Variantă Albă", "Dat în exploatare") feed the building-era mapper.
    """

    apartment_paths = ("/complexe-rezidentiale",)
    house_paths = ("/case",)

    def _from_scripts(
        self, soup: BeautifulSoup, property_type: str
    ) -> list[RawListing]:
        listings: list[RawListing] = []
        for script in soup.find_all("script"):
            raw = script.string or script.get_text() or ""
            raw = raw.strip()
            if not raw or "{" not in raw:
                continue
            payload = _extract_json(raw)
            if payload is None:
                continue
            for item in _iter_listing_objects(payload):
                listings.append(
                    self._raw_from_json(item, property_type)
                )
        return listings

    def _raw_from_json(
        self, item: dict[str, Any], property_type: str
    ) -> RawListing:
        def pick(*keys: str) -> str:
            for key in keys:
                if key in item and item[key] not in (None, ""):
                    return str(item[key])
            return ""

        return RawListing(
            source=self.source,
            property_type=property_type,
            title=pick("title", "name", "denumire"),
            price_raw=pick("price", "pret", "price_eur", "cost"),
            area_raw=pick("area", "suprafata", "sqm", "size"),
            land_raw=pick("land", "teren", "land_ari"),
            floor_raw=pick("floor", "etaj"),
            sector_raw=pick("sector", "zone", "district", "address", "adresa"),
            building_raw=pick(
                "construction_phase", "faza", "stage", "material", "tip"
            ),
            repair_raw=pick("finish", "repair", "reparatie", "stare"),
            house_type_raw=pick("house_type", "tip_casa", "type"),
            levels_raw=pick("levels", "niveluri", "etaje"),
            listed_at=pick("created_at", "date", "data"),
            url=pick("url", "link"),
        )

    def parse(self, html: str, property_type: str = APARTMENT) -> list[RawListing]:
        soup = BeautifulSoup(html or "", "html.parser")
        listings = self._from_scripts(soup, property_type)
        if listings:
            return listings

        # DOM fallback for server-rendered agency cards.
        for card in soup.select(
            ".property-card, .complex-card, .listing-card, article, .estate-item"
        ):
            features: dict[str, str] = {}
            for cell in card.select(".spec, .params li, li, .property-feature"):
                text = self._text(cell)
                if ":" in text:
                    label, _, value = text.partition(":")
                    features[_norm_text(label)] = value.strip()

            title = self._text(card.select_one(".title, h2, h3, a"))
            listings.append(
                RawListing(
                    source=self.source,
                    property_type=property_type,
                    title=title,
                    price_raw=self._text(
                        card.select_one(".price, .pret, .cost")
                    ),
                    area_raw=_first_feature(
                        features, ("suprafata", "area", "площадь")
                    ),
                    land_raw=_first_feature(features, ("teren", "land")),
                    floor_raw=_first_feature(features, ("etaj", "floor")),
                    sector_raw=_first_feature(
                        features, ("sector", "zona", "adresa")
                    )
                    or title,
                    building_raw=_first_feature(
                        features, ("faza", "constructie", "material", "tip")
                    )
                    or title,
                    repair_raw=_first_feature(
                        features, ("finisaj", "reparatie", "stare")
                    ),
                    house_type_raw=_first_feature(features, ("tip", "type"))
                    or title,
                    levels_raw=_first_feature(features, ("niveluri", "etaje")),
                    url=_first_href(card),
                )
            )
        return listings


class ProimobilScraper(_ResidentialComplexScraper):
    source = "proimobil.md"


class OximobilScraper(_ResidentialComplexScraper):
    source = "oximobil.md"


class LaraScraper(BaseScraper):
    """Lara.md - conservative appraisal baseline rendered as index tables.

    Data lives in classic ``<table>`` rows; the first row is treated as a
    header so columns can be addressed by name across slightly varying
    layouts. These rows anchor the regression against listing inflation.
    """

    source = "lara.md"
    apartment_paths = ("/",)
    house_paths = ("/",)

    _HEADER_MAP = {
        "pret": "price",
        "price": "price",
        "цена": "price",
        "suprafata": "area",
        "площадь": "area",
        "teren": "land",
        "участок": "land",
        "etaj": "floor",
        "этаж": "floor",
        "sector": "sector",
        "район": "sector",
        "tip": "building",
        "тип": "building",
        "reparatie": "repair",
        "ремонт": "repair",
        "niveluri": "levels",
    }

    def parse(self, html: str, property_type: str = APARTMENT) -> list[RawListing]:
        soup = BeautifulSoup(html or "", "html.parser")
        listings: list[RawListing] = []

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            header_cells = rows[0].find_all(["th", "td"])
            columns = [
                self._HEADER_MAP.get(_norm_text(self._text(c)), "")
                for c in header_cells
            ]
            if not any(columns):
                continue
            for tr in rows[1:]:
                cells = tr.find_all(["td", "th"])
                if not cells:
                    continue
                values = {
                    columns[i]: self._text(cells[i])
                    for i in range(min(len(columns), len(cells)))
                    if columns[i]
                }
                if not values:
                    continue
                listings.append(
                    RawListing(
                        source=self.source,
                        property_type=property_type,
                        title=values.get("building", ""),
                        price_raw=values.get("price", ""),
                        area_raw=values.get("area", ""),
                        land_raw=values.get("land", ""),
                        floor_raw=values.get("floor", ""),
                        sector_raw=values.get("sector", ""),
                        building_raw=values.get("building", ""),
                        repair_raw=values.get("repair", ""),
                        house_type_raw=values.get("building", ""),
                        levels_raw=values.get("levels", ""),
                        url=_first_href(tr),
                    )
                )
        return listings


# ---------------------------------------------------------------------------
# Scraper registry helpers
# ---------------------------------------------------------------------------
SCRAPERS: dict[str, BaseScraper] = {
    NineNineNineScraper.source: NineNineNineScraper(),
    MaklerScraper.source: MaklerScraper(),
    ProimobilScraper.source: ProimobilScraper(),
    OximobilScraper.source: OximobilScraper(),
    LaraScraper.source: LaraScraper(),
}


def get_scraper(source: str) -> BaseScraper:
    """Return the scraper registered for ``source`` (KeyError if unknown)."""
    return SCRAPERS[source]


def parse_all(
    payloads: Iterable[tuple[str, str, str]]
) -> list[NormalizedListing]:
    """Normalize a batch of ``(source, html, property_type)`` payloads."""
    results: list[NormalizedListing] = []
    for source, html, property_type in payloads:
        scraper = SCRAPERS.get(source)
        if scraper is None:
            continue
        results.extend(scraper.parse_normalized(html, property_type))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _first_feature(features: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        for label, value in features.items():
            if key in label:
                return value
    return ""


def _first_href(node: Any) -> str:
    if node is None:
        return ""
    anchor = node.find("a", href=True) if hasattr(node, "find") else None
    return anchor["href"] if anchor else ""


def _extract_json(raw: str) -> Optional[Any]:
    """Pull the first balanced JSON object/array out of a script body."""
    # Strip common "window.__X__ = {...};" assignment wrappers.
    eq = raw.find("=")
    candidates = [raw]
    if eq != -1:
        candidates.append(raw[eq + 1 :].strip().rstrip(";"))
    for candidate in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end != -1 and end > start:
                snippet = candidate[start : end + 1]
                try:
                    return json.loads(snippet)
                except (json.JSONDecodeError, ValueError):
                    continue
    return None


def _iter_listing_objects(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield listing-like dicts from arbitrarily nested JSON metadata."""
    if isinstance(payload, dict):
        # JSON-LD style.
        if payload.get("@type") in {"Offer", "Product", "Residence", "Apartment"}:
            yield payload
        for key in ("listings", "items", "data", "results", "offers", "@graph"):
            if key in payload:
                yield from _iter_listing_objects(payload[key])
        # A bare listing dict that has price-ish keys.
        if any(k in payload for k in ("price", "pret", "price_eur")):
            yield payload
    elif isinstance(payload, list):
        for element in payload:
            yield from _iter_listing_objects(element)

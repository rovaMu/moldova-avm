"""Unit tests for the multilingual normalizers on dirty/edge-case inputs."""

import pytest

from app.scraper import parser as p


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Centru", p.SECTOR_CENTRU),
        ("Центр", p.SECTOR_CENTRU),
        ("Râșcani", p.SECTOR_RASCANI),
        ("Rascani", p.SECTOR_RASCANI),
        ("Рышкановка", p.SECTOR_RASCANI),
        ("Botanica", p.SECTOR_BOTANICA),
        ("Ботаника", p.SECTOR_BOTANICA),
        ("Buiucani", p.SECTOR_BUIUCANI),
        ("Буюканы", p.SECTOR_BUIUCANI),
        ("Ciocana", p.SECTOR_CIOCANA),
        ("Чеканы", p.SECTOR_CIOCANA),
        ("Telecentru", p.SECTOR_TELECENTRU),
        ("Телецентр", p.SECTOR_TELECENTRU),
        ("Durlești", p.SECTOR_SUBURBS),
        ("Cricova", p.SECTOR_SUBURBS),
        ("Stăuceni", p.SECTOR_SUBURBS),
        # Fail-safe default -> Botanica
        ("", p.SECTOR_BOTANICA),
        (None, p.SECTOR_BOTANICA),
        ("Some unknown place", p.SECTOR_BOTANICA),
        ("   ", p.SECTOR_BOTANICA),
    ],
)
def test_normalize_sector(value, expected):
    assert p.normalize_sector(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Bloc nou", 1),
        ("Varianta albă", 1),
        ("Новострой", 1),
        ("Сдан в эксплуатацию", 1),
        ("Bloc vechi", 0),
        ("Hrușciovka", 0),
        ("Хрущевка", 0),
        ("Seria 143", 0),
        ("Apartament MS", 0),
        ("Moldovenească", 0),
        ("Вторичный рынок", 0),
        # new building keyword wins over old when both appear
        ("Bloc nou, fost vechi", 1),
        # fail-safe default -> 1
        ("", 1),
        (None, 1),
        ("random text", 1),
    ],
)
def test_normalize_building_era(value, expected):
    assert p.normalize_building_era(value) == expected


def test_ms_token_not_matched_inside_words():
    # "MS" should only match as a standalone Soviet tag, not inside words.
    assert p.normalize_building_era("Comsomolului street") == 1


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Fără reparație", 1),
        ("Variantă albă", 1),
        ("Без ремонта", 1),
        ("Черновая отделка", 1),
        ("Reparație cosmetică", 2),
        ("Косметический ремонт", 2),
        ("Жилое состояние", 2),
        ("Euroreparație", 3),
        ("Евроремонт", 3),
        ("Lux", 4),
        ("Design individual", 4),
        ("Эксклюзив", 4),
        # tier precedence: lux beats euro
        ("euro lux design", 4),
        # fail-safe default -> 3
        ("", 3),
        (None, 3),
        ("whatever", 3),
    ],
)
def test_normalize_repair_level(value, expected):
    assert p.normalize_repair_level(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("65 m²", 65.0),
        ("45.5 mp", 45.5),
        ("80,5 m2", 80.5),
        ("кв.м. 70", 70.0),
        ("Suprafața: 100 кв.м", 100.0),
        ("garbage", None),
        (None, None),
        (55, 55.0),
    ],
)
def test_extract_sqm(value, expected):
    assert p.extract_sqm(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("6 ari", 6.0),
        ("0.06 ha", 6.0),
        ("4.5 sote", 4.5),
        ("10 соток", 10.0),
        ("0,5 ha", 50.0),
        ("no land here", None),
        (None, None),
    ],
)
def test_extract_land_ari(value, expected):
    assert p.extract_land_ari(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("3/9", (3, 9)),
        ("5 de la 10", (5, 10)),
        ("5 din 10", (5, 10)),
        ("5 из 10", (5, 10)),
        ("7", (7, None)),
        ("nonsense", (None, None)),
        (None, (None, None)),
    ],
)
def test_split_floor(value, expected):
    assert p.split_floor(value) == expected


@pytest.mark.parametrize(
    "current,total,expected",
    [
        (1, 9, 0),      # ground floor penalty
        (9, 9, 0),      # top floor penalty
        (5, 9, 1),      # middle floor
        (None, 9, 1),   # missing -> no penalty default
        (5, None, 1),
        (5, 0, 1),
    ],
)
def test_compute_is_middle_floor(current, total, expected):
    assert p.compute_is_middle_floor(current, total) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Casă Individuală", p.HOUSE_INDIVIDUAL),
        ("Townhouse", p.HOUSE_TOWNHOUSE),
        ("Duplex", p.HOUSE_TOWNHOUSE),
        ("Vilă", p.HOUSE_VILLA),
        ("Casă de vacanță", p.HOUSE_VILLA),
        ("дача", p.HOUSE_VILLA),
        ("", p.HOUSE_INDIVIDUAL),
        (None, p.HOUSE_INDIVIDUAL),
    ],
)
def test_normalize_house_sub_type(value, expected):
    assert p.normalize_house_sub_type(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2 niveluri", 2),
        ("3 этажа", 3),
        ("P+1", 1),
        (None, None),
        ("none", None),
    ],
)
def test_extract_levels(value, expected):
    assert p.extract_levels(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("85 000 €", 85000.0),
        ("€1,250,000", 1250000.0),
        ("45000", 45000.0),
        ("1 EUR", 1.0),
        ("preț la telefon", None),
        (None, None),
    ],
)
def test_extract_price_eur(value, expected):
    assert p.extract_price_eur(value) == expected

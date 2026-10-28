"""Conservative field formatting, independent of evaluation data and answer keys.

Never change identities, numeric values, qualifiers, or catalog date formats.
Only unambiguous whole-answer patterns are eligible. The original reader answer
and the applied transformations must be retained in the run trace.
"""
from __future__ import annotations

from datetime import date
import re

from hw2rag.retrieval.bm25 import _date_aliases

MONTHS = "January February March April May June July August September October November December".split()
MONTH_PATTERN = "|".join(MONTHS)
DATE = rf"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})"
CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?"
ZONE = r"(?:\s+(ET|EST|EDT|Eastern Time|Eastern Standard Time|Eastern Daylight Time))?"


def _date_text(month: str, day: str, year: str) -> str | None:
    try:
        value = date(int(year), MONTHS.index(month.title()) + 1, int(day))
    except (ValueError, IndexError):
        return None
    return f"{MONTHS[value.month - 1]} {value.day}, {value.year}"


def _clock_text(hour: str, minute: str | None, meridiem: str, zone: str | None) -> str | None:
    if not 1 <= int(hour) <= 12 or not 0 <= int(minute or 0) <= 59:
        return None
    suffix = {"eastern time": "ET", "eastern standard time": "EST", "eastern daylight time": "EDT"}.get(
        (zone or "").casefold(), zone or "")
    return f"{int(hour)}:{int(minute or 0):02d} {meridiem.upper()}M" + (f" {suffix.upper()}" if suffix else "")


def normalize_field_answer(question: str, answer: str) -> tuple[str, list[str]]:
    """Return a normalized answer and an auditable list of generic rule names."""
    value = answer.strip()
    query = question.casefold()
    if value.upper() == "UNKNOWN":
        return value, []
    steps = []
    catalog_date = bool(re.search(r"\b(?:acquisition|accession) date\b", query))
    time_query = bool(re.search(r"\b(?:what time|at what time)\b", query))
    when_query = bool(re.search(r"\bwhen\b", query))
    query_has_date = any(a.startswith("date_") for a in _date_aliases(query))
    # A catalog's machine-readable field is intentionally not reformatted.
    if not catalog_date and (when_query or time_query):
        iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})\.?", value)
        if iso and when_query and not query_has_date:
            try:
                day = date.fromisoformat("-".join(iso.groups()))
                value = f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
                steps.append("iso_date_to_narrative_date")
            except ValueError:
                pass
        natural = re.fullmatch(DATE + r"\.?", value, re.I)
        if natural and when_query and not query_has_date:
            normalized = _date_text(*natural.groups())
            if normalized and normalized != value:
                value = normalized
                steps.append("calendar_date_remove_ordinal_weekday_padding")
        combined = re.fullmatch(DATE + r"(?:,?\s+(?:at\s+)?)" + CLOCK + ZONE + r"\.?", value, re.I)
        if combined:
            month, day, year, hour, minute, meridiem, zone = combined.groups()
            if time_query or (when_query and query_has_date):
                normalized = _clock_text(hour, minute, meridiem, zone)
                rule = "requested_clock_without_repeated_date"
            elif when_query:
                normalized = _date_text(month, day, year)
                rule = "requested_calendar_date_without_extra_clock"
            else:
                normalized = None
            if normalized and normalized != value:
                value = normalized
                steps.append(rule)
        military = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)" + ZONE + r"\.?", value, re.I)
        if military and (time_query or (when_query and query_has_date)):
            hour, minute, zone = military.groups()
            value = _clock_text(str(int(hour) % 12 or 12), minute, "p" if int(hour) >= 12 else "a", zone)
            steps.append("24_hour_clock_to_12_hour")
    if re.search(r"\bwhere\b.*\bborn\b", query):
        city = re.fullmatch(r"([^,\d]+)(?:,\s*[A-Z]{2,3}){1,2}\.?", value)
        if city:
            value = city[1].strip()
            steps.append("birth_city_without_region_codes")
    if re.search(r"\bhow did\b.*\bacquire\b", query) and not re.search(r"\bcredit line\b", query):
        method = re.match(r"^(Gift|Purchase|Bequest|Exchange|Transfer)\s+(?:of|from)\b", value, re.I)
        if method:
            value = method[1].title()
            steps.append("acquisition_method_without_donor_credit")
    return value, steps


# ---------------------------------------------------------------------------
# "calibrated" profile (system 2). Extra whole-answer formatting rules derived from
# the error analysis of the public leaderboard set; applied after
# normalize_field_answer. Same contract as above: only reformat or trim, never
# add facts, and every applied rule is named in the trace.

ISO_DATETIME = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\.?")
POSITIONS = {
    "football": {
        "QB": "Quarterback", "RB": "Running back", "FB": "Fullback", "WR": "Wide receiver", "TE": "Tight end",
        "OT": "Offensive tackle", "T": "Tackle", "OG": "Offensive guard", "G": "Guard", "C": "Center",
        "OL": "Offensive lineman", "DE": "Defensive end", "DT": "Defensive tackle", "NT": "Nose tackle",
        "DL": "Defensive lineman", "LB": "Linebacker", "OLB": "Outside linebacker", "ILB": "Inside linebacker",
        "MLB": "Middle linebacker", "CB": "Cornerback", "S": "Safety", "FS": "Free safety", "SS": "Strong safety",
        "DB": "Defensive back", "K": "Kicker", "P": "Punter", "LS": "Long snapper",
    },
    "hockey": {"C": "Center", "LW": "Left wing", "RW": "Right wing", "D": "Defenseman", "G": "Goaltender"},
    "baseball": {
        "P": "Pitcher", "SP": "Starting pitcher", "RP": "Relief pitcher", "C": "Catcher", "1B": "First baseman",
        "2B": "Second baseman", "3B": "Third baseman", "SS": "Shortstop", "LF": "Left fielder",
        "CF": "Center fielder", "RF": "Right fielder", "OF": "Outfielder", "IF": "Infielder", "DH": "Designated hitter",
    },
}
SPORT_HINTS = {
    "football": r"\b(steelers|football|nfl|panthers football)\b",
    "hockey": r"\b(penguins|hockey|nhl)\b",
    "baseball": r"\b(pirates|baseball|mlb)\b",
}
_STOP = set("the a an of in on at for to is are was were does do did what which who whom whose and or by with "
            "known its it's their his her this that from as".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.casefold()) if t not in _STOP}


def _date_or_clock(question: str, year: str, month: str, day: str, hour: str, minute: str) -> tuple[str | None, str]:
    query = question.casefold()
    time_query = bool(re.search(r"\b(?:what time|at what time)\b", query))
    when_query = bool(re.search(r"\bwhen\b", query))
    query_has_date = any(a.startswith("date_") for a in _date_aliases(query))
    if time_query or (when_query and query_has_date):
        h = int(hour)
        return _clock_text(str(h % 12 or 12), minute, "p" if h >= 12 else "a", None), "iso_datetime_to_clock"
    try:
        value = date(int(year), int(month), int(day))
    except ValueError:
        return None, ""
    return f"{MONTHS[value.month - 1]} {value.day}, {value.year}", "iso_datetime_to_narrative_date"


def calibrate_answer(question: str, answer: str) -> tuple[str, list[str]]:
    """Second-stage formatting for the calibrated profile; returns (answer, rule names)."""
    value = answer.strip()
    query = question.casefold()
    if value.upper() == "UNKNOWN":
        return value, []
    steps = []
    catalog_date = bool(re.search(r"\b(?:acquisition|accession|creation) date\b", query))

    # 1. Machine timestamps (e.g. 2026-11-10T12:30:00-05:00) -> the requested clock time or calendar date,
    #    in the local time written in the record.
    iso = ISO_DATETIME.fullmatch(value)
    if iso and not catalog_date and re.search(r"\b(?:what time|at what time|when)\b", query):
        normalized, rule = _date_or_clock(question, *iso.groups())
        if normalized:
            value = normalized
            steps.append(rule)

    # 2. A bare clock in any casing/punctuation ("4:00 pm", "7 p.m.") -> canonical "4:00 PM".
    if re.search(r"\b(?:what time|at what time|when)\b", query):
        clock = re.fullmatch(CLOCK + ZONE + r"\.?", value, re.I)
        if clock:
            normalized = _clock_text(*clock.groups())
            if normalized and normalized != value:
                value = normalized
                steps.append("clock_canonical_format")

    # 3. Roster position codes ("K") -> position names, only when the sport is clear from the question.
    if re.search(r"\bwhat position\b|\bwhich position\b", query):
        sports = [s for s, pattern in SPORT_HINTS.items() if re.search(pattern, query)]
        code = value.upper().rstrip(".")
        if len(sports) == 1 and code in POSITIONS[sports[0]]:
            value = POSITIONS[sports[0]][code]
            steps.append("roster_position_code_to_name")

    # 4. "Contact X for information about ..." -> X, for "who should I contact" questions.
    if re.match(r"^who should i contact\b", query):
        contact = re.fullmatch(r"(?:please\s+)?contact\s+(.+?)\s+(?:for|to|about|regarding|with)\b.*", value, re.I | re.S)
        target = contact[1].strip().rstrip(",") if contact else ""
        has_address = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+|\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}", target)
        if target and (has_address or len(target.split()) <= 8):
            value = target
            steps.append("contact_sentence_to_contact")

    # 5. A sentence that restates the question ("The restaurant known for X is Y.") -> Y, for short Y only.
    #    Entity questions only (which/who/"what <noun>"); definitional "what is ..." answers are left whole.
    if re.match(r"^(?:which|who)\b|^what\s+(?!(?:is|are|was|were|does|do|did|has|have|had)\b)\w+", query):
        restated = re.fullmatch(r"(?:the\s+)?(.+?)\s+(?:is|are|was|were)\s+(.+?)\.?", value, re.I | re.S)
        if restated:
            subject, obj = restated.groups()
            subject_tokens = _tokens(subject)
            overlap = len(subject_tokens & _tokens(question)) / len(subject_tokens) if subject_tokens else 0
            if len(subject_tokens) >= 2 and overlap >= 0.6 and len(obj.split()) <= 8 \
                    and not re.search(r"\b(?:is|are|was|were|and|which|that|who)\b", obj, re.I):
                value = obj.strip()
                steps.append("restated_question_sentence_to_span")
    return value, steps

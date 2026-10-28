"""Opt-in record scoping and full-snapshot aggregation, without answer keys.

All fields come from exact indexed source spans. Counts are labelled derived,
retain membership and duplicate identities, and never treat top-k as exhaustive.
The v1 field-aware implementation is intentionally left unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import re
from urllib.parse import urlsplit

from hw2rag.retrieval.field_evidence import FieldEvidence, FIELD_STOP, words
from hw2rag.retrieval.focused import MONTHS, folded
from hw2rag.retrieval.bm25 import _date_aliases

VERSION = "verified_record_scopes_v2"


def quoted_titles(query: str) -> set[str]:
    return {folded(a or b) for a, b in re.findall(r'“([^”]+)”|"([^"]+)"', query)}


def requested_catalog_field(query: str) -> str | None:
    q = query.casefold()
    for phrase, label in (("accession", "Accession number"), ("credit line", "Credit line"),
                          ("acquisition date", "Acquisition date"), ("creation date", "Creation date"),
                          ("department", "Department")):
        if phrase in q:
            return label
    if re.search(r"\b(?:artist|school|creator|attribution|painted)\b", q):
        return "Artist"
    if "acquire" in q or "acquisition method" in q:
        return "Acquisition method"
    if "currently" in q and ("display" in q or "list" in q):
        return "Current location"
    if "materials" in q or "medium" in q:
        return "Medium"
    return None


def round_words(value: str) -> set[str]:
    value = re.sub(r"\bnational league\b", "NL", value, flags=re.I)
    value = re.sub(r"\bamerican league\b", "AL", value, flags=re.I)
    return words(value)


class RecordSafeEvidence(FieldEvidence):
    def _scoped_view(self, document_id: str, body: str, spans: list[dict], operation: dict) -> dict:
        view = self._view(document_id, body, spans, "record_scope", operation)
        view["chunk"]["evidence_view"]["version"] = VERSION
        return view

    def _catalog_views(self, query: str) -> list[dict]:
        if re.search(r"\b(?:concert|performance|venue)\b", query, re.I):
            return []
        titles, label = quoted_titles(query), requested_catalog_field(query)
        if not titles or not label:
            return []
        matches = [d for d in self.documents.values() if d["document_type"] == "artwork" and folded(d["title"]) in titles]
        views = []
        for document in matches:
            selected = list(re.finditer(rf"(?m)^(?:Title|Artist|Accession number|{re.escape(label)}):[^\n]+", document["text"]))
            target = next((m for m in selected if m[0].startswith(label + ":")), None)
            if target is None:
                return []  # A missing field is not an implicit empty/negative value.
            spans = [self._span(document["document_id"], m.start(), m.end()) for m in selected]
            body = (f"Exact-title catalog record. Requested field: {label}.\n"
                    "Read this record only; other similarly named works are not this object.\n"
                    + "\n".join(span["text"] for span in spans))
            if len(matches) > 1:
                body += "\nScope warning: multiple exact-title records exist; the title alone does not distinguish their object IDs."
            views.append(self._scoped_view(document["document_id"], body, spans,
                {"operation": "project_exact_title_catalog_field", "field": label, "exact_title_record_count": len(matches)}))
        return views

    def _event_role_views(self, query: str) -> list[dict]:
        titles = quoted_titles(query)
        role = next((r for r in ("conductor", "composer", "soloist", "director") if re.search(rf"\b{r}\b", query, re.I)), None)
        if not titles or not role:
            return []
        dates = {a for a in _date_aliases(query.casefold()) if a.startswith("date_")}
        views = []
        for doc in self.documents.values():
            if doc["document_type"] != "event" or folded(doc["title"]) not in titles:
                continue
            temporal = doc.get("temporal", {})
            day = str(temporal.get("local_date") or temporal.get("local_start") or "")[:10]
            if dates and "date_" + day.replace("-", "_") not in dates:
                continue
            present = bool(re.search(rf"\b{role}\b", doc["text"], re.I))
            body = (f"Exact-title event record. Requested role: {role}.\n"
                    "An institution-wide leader is not necessarily the person in this role for this event.\n"
                    + ("Requested role: NOT SPECIFIED in this exact event record.\n" if not present else "") + doc["text"])
            views.append(self._scoped_view(doc["document_id"], body,
                [self._span(doc["document_id"], 0, len(doc["text"]))],
                {"operation": "scope_exact_event_role", "role": role, "role_label_present": present}))
        return views

    def _millage_view(self, query: str) -> dict | None:
        if not re.search(r"\b(?:real estate|property) tax rate\b", query, re.I):
            return None
        entity = words(query) - FIELD_STOP - {"real", "estate", "property", "rate", "municipality", "borough", "city", "township", "municipal"}
        entity = {w for w in entity if not w.isdigit()}
        if not entity:
            return None
        requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        if len(requested_years) > 1:
            return None
        choices = []
        for doc in self.documents.values():
            text = doc["text"]
            header = re.search(r"MILLAGE\s+((?:19|20)\d{2})\s+MILLAGE\s+((?:19|20)\d{2})", text, re.I)
            if not header or not re.search(r"\bMILL\b", text, re.I):
                continue
            years = list(header.groups())
            year = next(iter(requested_years)) if requested_years else max(years)
            if year not in years:
                continue
            pattern = r"(?m)^(.+?)\s+\d{3}[-.]\d{3}[-.]\d{4}(?:\s*(?:x|ext\.?)\s*\d+)?\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[\d,]+\$?\s*$"
            matching = [m for m in re.finditer(pattern, text) if entity <= words(m[1])]
            if len(matching) != 1:
                continue
            row = matching[0]
            value = row[2 + years.index(year)]
            # Unit expansion is explicit; numeric cells and source year order
            # are copied, not inferred from unrelated percentages.
            unit = re.search(r"\bMILL\b", text, re.I)
            spans = [self._span(doc["document_id"], header.start(), header.end()),
                     self._span(doc["document_id"], unit.start(), unit.end()),
                     self._span(doc["document_id"], row.start(), row.end())]
            body = (f"Requested property-tax millage, including its unit: {value} mills\n"
                f"Selected source year column: {year}. Source term MILLAGE is expressed in mills, not percent.\n"
                "Source column headings and exact municipality row:\n" + "\n".join(s["text"] for s in spans))
            choices.append(self._scoped_view(doc["document_id"], body, spans,
                {"operation": "project_explicit_millage_column", "year": year, "source_year_order": years, "value": value, "unit": "mills"}))
        values = {v["chunk"]["evidence_view"]["operations"]["value"] for v in choices}
        return choices[0] if len(values) == 1 else None

    def _threshold_view(self, query: str) -> dict | None:
        if not re.search(r"\b(?:minimum|threshold)\b", query, re.I) or not re.search(r"\bincome\b", query, re.I):
            return None
        entity = words(query) - FIELD_STOP - {"income", "lst", "earned", "net", "profits", "sources", "total", "local", "services", "municipality", "township", "city", "borough", "county"}
        if not entity:
            return None
        choices = []
        for doc in self.documents.values():
            if doc["document_type"] not in {"web_page", "pdf_page"} or not entity <= words(doc["title"] + " " + doc["text"][:900]):
                continue
            text = doc["text"]
            for match in re.finditer(r"\b(less than|below|under)\s+(\$\s*\d[\d,]*(?:\.\d+)?)", text, re.I):
                start, end = max(0, match.start() - 180), min(len(text), match.end() + 180)
                context = text[start:end]
                if not re.search(r"\bincome\b", context, re.I):
                    continue
                comparison = " ".join(match[0].split())
                body = (f"Requested income-eligibility comparison INCLUDING its operator: {comparison}\n"
                    "The boundary amount alone loses the strict comparison. Retain any additional source conditions.\nSource context (verbatim):\n" + context)
                choices.append(self._scoped_view(doc["document_id"], body,
                    [self._span(doc["document_id"], start, end)],
                    {"operation": "project_explicit_income_comparison", "comparator": match[1].casefold(), "amount": match[2]}))
        values = {(v["chunk"]["evidence_view"]["operations"]["comparator"], v["chunk"]["evidence_view"]["operations"]["amount"]) for v in choices}
        return choices[0] if len(values) == 1 else None

    def _schedule_groups(self) -> dict[tuple, list[dict]]:
        groups = defaultdict(list)
        for doc in self.documents.values():
            if doc["document_type"] != "game":
                continue
            attr, temporal = doc.get("attributes", {}), doc.get("temporal", {})
            date = temporal.get("local_date")
            if not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                continue
            if attr.get("source_role") == "structured_schedule_record" and attr.get("team") and attr.get("sport"):
                clock = str(temporal.get("local_start") or "")[11:16]
                opponent = str(attr.get("opponent") or "")
                if not opponent:
                    continue
                team, sport, season = attr["team"], attr["sport"], str(attr.get("season") or "")
                identity = ("schedule", folded(team), folded(sport), season, date, folded(opponent), clock)
                groups[(folded(team), folded(sport), season)].append({"doc": doc, "identity": identity,
                    "team": team, "sport": sport, "date": date, "opponent": opponent,
                    "home": attr.get("home_away_marker") == "vs"})
            elif attr.get("home_team") and attr.get("away_team") and str(doc.get("external_id", "")).isdigit():
                for side, other in (("home_team", "away_team"), ("away_team", "home_team")):
                    team = attr[side]
                    groups[(folded(team), folded(attr.get("sport", "")), str(attr.get("season") or ""))].append({
                        "doc": doc, "identity": ("source_game_id", urlsplit(doc.get("canonical_url", "")).netloc.removeprefix("www."), str(doc["external_id"])),
                        "team": team, "sport": attr.get("sport", ""), "date": date,
                        "opponent": attr[other], "home": side == "home_team"})
        return groups

    def _schedule_count(self, query: str) -> dict | None:
        if re.search(r"\b(?:regular season|preseason|postseason|exhibition|playoffs?|cancelled|canceled|postponed)\b", query, re.I):
            return None  # Unsupported competition/status qualifiers cannot become all-phase counts.
        if not re.search(r"\bhow many games\b", query, re.I) or re.search(r"\b(?:won|win|lost|loss)\b", query, re.I):
            return None
        month = next(((i + 1, y) for i, name in enumerate(MONTHS)
            for y in re.findall(rf"\b{name}\s+(20\d{{2}})\b", query, re.I)), None)
        if month is None:
            return None
        qwords = words(query)
        choices = []
        for (team, sport, season), records in self._schedule_groups().items():
            entity = words(team) - FIELD_STOP
            sports = words(sport) & {"mens", "womens", "basketball", "volleyball", "football", "soccer", "hockey", "baseball"}
            if not entity or not entity <= qwords or not sports <= qwords:
                continue
            prefix = f"{month[1]}-{month[0]:02d}-"
            selected = [r for r in records if r["date"].startswith(prefix)
                and ("home" not in qwords or r["home"]) and ("away" not in qwords or not r["home"])]
            if selected:
                choices.append((len(entity), selected))
        if not choices:
            return None
        best_score = max(score for score, _ in choices)
        selected_groups = [rs for score, rs in choices if score == best_score]
        if len(selected_groups) != 1:
            return None  # Ambiguous team/sport/season is not a count.
        records = selected_groups[0]
        distinct = {r["identity"]: r for r in records}
        # Missing display times can be duplicate or a doubleheader: do not guess.
        slots = defaultdict(set)
        for identity in distinct:
            if identity[0] == "schedule":
                slots[identity[:-1]].add(identity[-1])
        if any("" in clocks and len(clocks) > 1 for clocks in slots.values()):
            return None
        members = sorted(distinct.values(), key=lambda r: (r["date"], r["identity"]))
        parent = members[0]["doc"]
        source_records = [{"document_id": r["doc"]["document_id"], "document_version_id": r["doc"]["version_id"],
            "identity": list(r["identity"]), "span": self._span(r["doc"]["document_id"], 0, len(r["doc"]["text"]))} for r in members]
        body = ("DERIVED aggregate, not a verbatim source number.\n"
            f"Team: {members[0]['team']}\nSport: {members[0]['sport']}\nCalendar month: {month[1]}-{month[0]:02d}\n"
            f"Distinct matching games in complete indexed schedule-record set: {len(members)}\n"
            "Each stable game identity is counted once, irrespective of repeated pages/chunks.\n"
            "Scope: all matching structured game records in the verified index snapshot, not an arbitrary top-k sample.\nMembers:\n"
            + "\n".join(f"{r['date']} | {r['opponent']} | {'home' if r['home'] else 'away'}" for r in members))
        view = self._scoped_view(parent["document_id"], body, [source_records[0]["span"]],
            {"operation": "count_distinct_schedule_records", "count": len(members), "team": members[0]["team"],
             "sport": members[0]["sport"], "calendar_month": f"{month[1]}-{month[0]:02d}",
             "duplicate_records_removed": len(records) - len(members), "index_document_count": len(self.documents)})
        view["chunk"]["evidence_view"]["source_records"] = source_records
        return view

    def _series_count(self, query: str) -> dict | None:
        if not re.search(r"\bhow many\b", query, re.I) or not re.search(r"\b(?:won|win)\b", query, re.I) or not re.search(r"\bseries\b", query, re.I) or re.search(r"\bgames?\b", query, re.I):
            return None
        qwords = round_words(query)
        choices = []
        for table in self.tables:
            headers = [words(h) for h in table.headers]
            rc = next((i for i, h in enumerate(headers) if h & {"round", "stage"}), None)
            wc = next((i for i, h in enumerate(headers) if "result" in h), None)
            if rc is None or wc is None:
                continue
            rows = [r for r in table.rows if len(round_words(r.cells[rc])) >= 2 and round_words(r.cells[rc]) <= qwords]
            if not rows or any(re.search(r"\bgame\b", r.cells[wc], re.I) for r in rows):
                continue
            doc = self.documents[table.document_id]
            round_terms = set().union(*(round_words(r.cells[rc]) for r in rows))
            entity = qwords - FIELD_STOP - round_terms - {"series"}
            entity_score = len(entity & words(doc["title"] + " " + str(doc.get("canonical_url", ""))))
            if not entity_score:
                continue
            won = [r for r in rows if re.match(r"(?:won|win|w)\b", r.cells[wc], re.I)]
            spans = [self._span(table.document_id, table.header_start, table.header_end)] + [self._span(table.document_id, r.start, r.end) for r in rows]
            body = (f"DERIVED distinct winning series rows: {len(won)}\n"
                "Unit: series, not individual games or series appearances. Duplicate source rows count once.\n"
                + "\n".join("; ".join(f"{h}: {v}" for h, v in zip(table.headers, r.cells)) for r in rows))
            choices.append((entity_score, self._scoped_view(table.document_id, body, spans,
                {"operation": "count_distinct_series_results", "count": len(won), "headers": list(table.headers),
                 "rows": [list(r.cells) for r in rows], "duplicate_rows_removed": table.duplicate_rows})))
        choices.sort(key=lambda x: -x[0])
        if not choices or len(choices) > 1 and choices[0][0] == choices[1][0]:
            return None
        return choices[0][1]

    def apply(self, query: str, results: list[dict], *, top_k: int) -> list[dict]:
        catalog = self._catalog_views(query)
        if catalog and len(catalog) <= top_k:
            return [{**view, "rank": rank} for rank, view in enumerate(catalog, 1)]
        event = self._event_role_views(query)
        if event and len(event) <= top_k:
            return [{**view, "rank": rank} for rank, view in enumerate(event, 1)]
        field = self._millage_view(query) or self._threshold_view(query)
        if field:
            return [field]
        count = self._schedule_count(query) or self._series_count(query)
        if count:
            return [count]
        return super().apply(query, results, top_k=top_k)

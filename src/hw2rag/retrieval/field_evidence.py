"""Query-time table projections and policy excerpts from verified documents.

No evaluation files, answer keys, entity maps, network calls, or generated
knowledge. Every projection retains exact source spans and indexed passage IDs.
Derived season end years / distinct counts are explicitly labelled operations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urldefrag

from hw2rag.retrieval.focused import folded, STOP

VERSION = "verified_field_evidence_v1"
FIELD_STOP = STOP | set("are was were has have been did will would should may could must be not any all many much during years year games game win won wins championship championships ranking rank strength schedule tax subject allowed received person partner self employed performances performance serve minimum exemption".split())


def words(value: str) -> set[str]:
    aliases = {"photographic": "photo", "photographs": "photo", "photography": "photo", "photos": "photo", "championships": "champion", "championship": "champion", "champions": "champion"}
    return {aliases.get(word, word) for word in folded(value).split()}


def season_end(value: str) -> int | None:
    match = re.fullmatch(r"((?:19|20|21)\d{2})\s*[-–/]\s*(\d{2}|\d{4})", value)
    if not match:
        return int(value) if re.fullmatch(r"(?:19|20|21)\d{2}", value) else None
    first, last = int(match[1]), int(match[2])
    if len(match[2]) == 2:
        last += first // 100 * 100
        if last < first:
            last += 100
    return last if 0 <= last - first <= 1 else None


@dataclass
class SourceRow:
    cells: tuple[str, ...]
    start: int
    end: int


@dataclass
class SourceTable:
    document_id: str
    headers: tuple[str, ...]
    header_start: int
    header_end: int
    rows: list[SourceRow] = field(default_factory=list)
    duplicate_rows: int = 0


def parse_tables(document: dict) -> list[SourceTable]:
    tables = []
    active = None
    offset = 0
    seen = set()
    for line in document["text"].splitlines(keepends=True):
        start, end = offset, offset + len(line.rstrip("\r\n"))
        offset += len(line)
        if "|" not in line:
            if line.strip():
                active = None
            continue
        cells = tuple(cell.strip() for cell in line.strip().split("|"))
        if cells and re.fullmatch(r"rank\s+team", cells[0], re.I):
            cells = ("Rank", "Team", *cells[1:])
        if active and "rank" in words(active.headers[0]) and len(cells) == len(active.headers) - 1:
            merged = re.fullmatch(r"(\d+)\s*([A-Za-z].+)", cells[0])
            if merged:
                cells = (*merged.groups(), *cells[1:])
        is_header = bool(words(" ".join(cells)) & {"rank", "year", "season", "result", "champion"}) and not re.match(r"^\d", cells[0])
        if is_header:
            active = SourceTable(document["document_id"], cells, start, end)
            tables.append(active)
            seen = set()
        elif active and len(cells) == len(active.headers) and re.match(r"^\d", cells[0]):
            key = tuple(map(folded, cells))
            if key in seen:
                active.duplicate_rows += 1
            else:
                seen.add(key)
                active.rows.append(SourceRow(cells, start, end))
        else:
            active = None
    return [table for table in tables if table.rows]


def load_verified_documents(corpus) -> dict[str, dict]:
    manifest = json.loads(corpus.manifest_path.read_text())
    relative = Path(manifest["input_document_manifest"])
    path = next((parent / relative for parent in corpus.manifest_path.parents if (parent / relative).is_file()), None)
    if path is None or sha256(path.read_bytes()).hexdigest() != manifest["input_document_manifest_sha256"]:
        raise ValueError("full document manifest differs from indexed snapshot")
    wanted = {c["document_id"]: c["document_version_id"] for c in corpus.chunks}
    documents = {}
    for shard in json.loads(path.read_text())["sources"]:
        if shard["source_id"] not in corpus.source_ids:
            continue
        data_path = (path.parent / shard["output_path"]).resolve()
        if not data_path.is_relative_to(path.parent.resolve()):
            raise ValueError("document shard escapes verified directory")
        content = data_path.read_bytes()
        if sha256(content).hexdigest() != shard["output_sha256"]:
            raise ValueError("document shard hash differs")
        for line in content.decode().splitlines():
            document = json.loads(line)
            identifier = document["document_id"]
            if identifier in wanted:
                if document["version_id"] != wanted[identifier]:
                    raise ValueError("document version differs from indexed chunks")
                documents[identifier] = document
    if documents.keys() != wanted.keys():
        raise ValueError("some indexed documents are missing")
    return documents


class FieldEvidence:
    def __init__(self, chunks: list[dict], documents: dict[str, dict]):
        self.chunks = chunks
        self.documents = documents
        self.by_document = defaultdict(list)
        for row, chunk in enumerate(chunks):
            self.by_document[chunk["document_id"]].append(row)
        self.tables = [table for d in documents.values() if d["document_type"] in {"web_page", "pdf_page"} for table in parse_tables(d)]

    @classmethod
    def load(cls, corpus):
        return cls(corpus.chunks, load_verified_documents(corpus))

    def _span(self, document_id: str, start: int, end: int) -> dict:
        document = self.documents[document_id]
        identifiers = [self.chunks[row]["chunk_id"] for row in self.by_document[document_id]
                       if self.chunks[row]["start_char"] < end and self.chunks[row]["end_char"] > start]
        if not identifiers:
            raise ValueError("source span is outside indexed passages")
        return {"start_char": start, "end_char": end, "text": document["text"][start:end], "chunk_ids": identifiers}

    def _view(self, document_id: str, body: str, spans: list[dict], kind: str, operations: dict) -> dict:
        identifiers = list(dict.fromkeys(identifier for span in spans for identifier in span["chunk_ids"]))
        row = next(row for row in self.by_document[document_id] if self.chunks[row]["chunk_id"] == identifiers[0])
        chunk = self.chunks[row]
        return {"rank": 1, "score": 100.0, "row_index": row, "component_scores": {"field_view": 100.0}, "component_ranks": {},
            "chunk": {**chunk, "retrieval_text": body, "context_chunk_ids": identifiers,
                "evidence_view": {"version": VERSION, "kind": kind, "document_id": document_id,
                    "document_version_id": chunk["document_version_id"], "source_spans": spans, "operations": operations}}}

    def _table_view(self, query: str) -> dict | None:
        qwords = words(query)
        ranking = bool(qwords & {"rank", "ranking"})
        championships = "years" in qwords and "champion" in qwords
        counting = "how many" in query.casefold() and bool(qwords & {"won", "win", "wins"})
        if not (ranking or championships or counting):
            return None
        subject = qwords - FIELD_STOP - {"champion", "tournament", "regular", "season"}
        subject = {w for w in subject if not w.isdigit()}
        choices = []
        for table in self.tables:
            headers = [words(h) for h in table.headers]
            if ranking:
                if not any("rank" in h for h in headers):
                    continue
                hits = [(len(subject & words(" ".join(row.cells[1:]))), row) for row in table.rows]
                hits = [(score, row) for score, row in hits if score]
                if not hits:
                    continue
                best_score = max(score for score, row in hits)
                rows = [row for score, row in hits if score == best_score]
                if len(rows) != 1:
                    continue
                score = 10 * best_score
                body = "Source table row with explicit column labels:\n" + "\n".join(f"{header}: {cell}" for header, cell in zip(table.headers, rows[0].cells))
                operations = {"operation": "select_matching_rank_row", "headers": list(table.headers), "rows": [list(rows[0].cells)]}
            elif championships:
                columns = [(len(h & qwords), i) for i, h in enumerate(headers) if "champion" in h]
                if not columns:
                    continue
                columns.sort(reverse=True)
                if len(columns) > 1 and columns[0][0] == columns[1][0]:
                    continue
                column = columns[0][1]
                hits = [(len(subject & words(row.cells[column])), row) for row in table.rows]
                best_score = max((score for score, row in hits), default=0)
                rows = [row for score, row in hits if score == best_score and score > 0]
                year_column = next((i for i, h in enumerate(headers) if h & {"year", "season"}), None)
                if not rows or year_column is None or any(season_end(row.cells[year_column]) is None for row in rows):
                    continue
                years = sorted({season_end(row.cells[year_column]) for row in rows})
                score = best_score * 10 + columns[0][0]
                body = f"Selected column: {table.headers[column]}\nDerived distinct championship calendar years (season end years): " + ", ".join(map(str, years)) + "\nSource rows:\n"
                body += "\n".join(f"Calendar year {season_end(row.cells[year_column])}; source season {row.cells[year_column]}; {table.headers[column]}: {row.cells[column]}" for row in rows)
                operations = {"operation": "filter_champion_column_and_normalize_season_end", "selected_column": table.headers[column], "headers": list(table.headers), "rows": [list(row.cells) for row in rows], "derived_years": years}
            else:
                round_column = next((i for i, h in enumerate(headers) if h & {"round", "stage"}), None)
                result_column = next((i for i, h in enumerate(headers) if "result" in h), None)
                if round_column is None or result_column is None:
                    continue
                rows = [row for row in table.rows if len(words(row.cells[round_column])) >= 2 and words(row.cells[round_column]) <= qwords]
                if not rows or not all(re.search(r"\bgame\b", row.cells[result_column], re.I) for row in rows):
                    continue
                won = [row for row in rows if re.match(r"(?:won|win|w)\b", row.cells[result_column], re.I)]
                score = max(len(words(row.cells[round_column])) for row in rows) * 10
                body = f"Distinct winning game rows: {len(won)}\nEach source row is counted once; repeated source/chunk copies are not additional games.\n"
                body += "\n".join("; ".join(f"{h}: {c}" for h, c in zip(table.headers, row.cells)) for row in rows)
                operations = {"operation": "count_distinct_explicit_game_results", "headers": list(table.headers), "rows": [list(row.cells) for row in rows], "winning_row_count": len(won), "duplicate_rows_removed": table.duplicate_rows}
            document = self.documents[table.document_id]
            score += len(subject & words(document["title"] + " " + (document.get("canonical_url") or "")))
            years_in_query = set(re.findall(r"\b20\d{2}\b", query))
            if years_in_query:
                score += 3 * len(years_in_query & set(re.findall(r"20\d{2}", document.get("canonical_url") or "")))
            spans = [self._span(table.document_id, table.header_start, table.header_end)] + [self._span(table.document_id, row.start, row.end) for row in rows]
            choices.append((score, self._view(table.document_id, body, spans, "table", operations)))
        return max(choices, key=lambda choice: choice[0])[1] if choices else None

    def _policy_views(self, query: str, results: list[dict]) -> list[dict]:
        if not re.search(r"\b(?:allowed|subject|exempt(?:ion)?|permission|prohibited|required)\b", query, re.I):
            return []
        if not re.match(r"^(?:is|are|can|may|do|does|must|should)\b", query, re.I) and not re.search(r"\b(?:minimum|threshold)\b", query, re.I):
            return []
        qwords = words(query)
        proper = set().union(*(words(token) for token in re.findall(r"\b[A-Z][\w’'-]+", query)))
        topics = qwords - FIELD_STOP - proper - {"policy", "policies"}
        conditions = re.compile(r"\b(?:not|excluding|except|unless|without|prohibited|required|permission|discretion|less than|below|exempt)\b", re.I)
        choices = []
        for document_id in dict.fromkeys(r["chunk"]["document_id"] for r in results):
            document = self.documents[document_id]
            if document["document_type"] not in {"pdf_page", "web_page"}:
                continue
            candidates = []
            for match in re.finditer(r"[^.!?]+(?:[.!?](?=\s|$)|$)", document["text"], re.S):
                text = match[0].strip()
                overlap = topics & words(text)
                if not overlap or not conditions.search(text) or len(text) > 1700:
                    continue
                score = len(overlap) * 5 + min(len(conditions.findall(text)), 3)
                candidates.append((score, match.start(), match.end(), len(overlap)))
            if not candidates:
                continue
            candidates.sort(reverse=True)
            strongest = max(candidate[3] for candidate in candidates)
            selected = [candidate for candidate in candidates if candidate[3] == strongest][:3]
            spans = [self._span(document_id, start, end) for _, start, end, _ in selected]
            # Keep an exception inside its FAQ question's scope, rather than
            # importing a different question's media/press-pass rules.
            headings = list(re.finditer(r"(?m)^\+\s+.+\?\s*$", document["text"]))
            for i, heading in enumerate(headings):
                block_end = headings[i + 1].start() if i + 1 < len(headings) else len(document["text"])
                if heading.start() <= selected[0][1] < block_end and block_end - heading.start() < 1800:
                    spans = [spans[0], self._span(document_id, heading.start(), block_end)]
                    break
            if re.search(r"\b(?:minimum|threshold)\b", query, re.I) and not any(re.search(
                    r"\b(?:less than|below|under)\s*\$?\s*\d", " ".join(span["text"].split()), re.I) for span in spans):
                continue
            body = "Relevant source conditions and exceptions (verbatim excerpts; retain all qualifiers):\n"
            body += "\n\n".join(" ".join(span["text"].split()) for span in spans)
            choices.append((selected[0][0], self._view(document_id, body, spans, "policy", {"operation": "select_query_relevant_conditional_sentences"})))
        choices.sort(key=lambda choice: -choice[0])
        return [view for score, view in choices[:2]]

    def apply(self, query: str, results: list[dict], *, top_k: int) -> list[dict]:
        table = self._table_view(query)
        views = [table] if table else self._policy_views(query, results)
        if not views:
            return results
        replaced = {urldefrag(view["chunk"].get("canonical_url") or view["chunk"]["document_id"])[0] for view in views}
        output = views + [result for result in results if urldefrag(result["chunk"].get("canonical_url") or result["chunk"]["document_id"])[0] not in replaced]
        return [{**result, "rank": rank} for rank, result in enumerate(output[:top_k], 1)]

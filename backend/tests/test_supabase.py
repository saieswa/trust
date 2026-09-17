"""Tests for Supabase metadata persistence using a fake client."""

from __future__ import annotations

from app.database.supabase import SupabaseDatabase


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, tables, name):
        self.tables = tables
        self.name = name
        self.filters = {}
        self.payload = None

    def insert(self, payload):
        self.payload = payload
        return self

    def select(self, _columns):
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def execute(self):
        if self.payload is not None:
            records = self.payload if isinstance(self.payload, list) else [self.payload]
            self.tables[self.name].extend(records)
            return FakeResponse(records)
        records = [
            record
            for record in self.tables[self.name]
            if all(record.get(key) == value for key, value in self.filters.items())
        ]
        return FakeResponse(records)


class FakeClient:
    def __init__(self):
        self.tables = {"documents": [], "chunks": []}

    def table(self, name):
        return FakeQuery(self.tables, name)


def test_inserts_and_retrieves_research_paper_metadata() -> None:
    database = SupabaseDatabase(client=FakeClient())
    document = database.insert_document(
        document_id="paper-attention",
        filename="attention_is_all_you_need.pdf",
        file_type="pdf",
        source="uploads/attention_is_all_you_need.pdf",
    )
    chunks = database.insert_chunks(
        [
            {
                "chunk_id": "paper-attention::chunk-0000",
                "document_id": "paper-attention",
                "filename": "attention_is_all_you_need.pdf",
                "page_number": 1,
                "text": "Attention is all you need.",
                "source": "attention_is_all_you_need.pdf#page=1",
                "metadata": {"char_count": 26},
            }
        ]
    )

    assert document["document_id"] == "paper-attention"
    assert document["processing_status"] == "uploaded"
    assert chunks[0]["chunk_id"] == "paper-attention::chunk-0000"
    assert database.get_document("paper-attention")["filename"] == "attention_is_all_you_need.pdf"
    assert database.get_chunks("paper-attention")[0]["page_number"] == 1


def test_missing_supabase_configuration_fails_without_client() -> None:
    database = SupabaseDatabase(url="", key="")
    try:
        database.client
    except RuntimeError as exc:
        assert "SUPABASE_URL" in str(exc)
    else:
        raise AssertionError("Expected missing Supabase configuration to fail")
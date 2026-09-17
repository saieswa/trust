"""Supabase access layer for persistent document and chunk metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.config import settings


class SupabaseDatabaseError(RuntimeError):
	"""Raised when Supabase configuration or a database operation fails."""


def _record_error(operation: str, exc: Exception) -> SupabaseDatabaseError:
	return SupabaseDatabaseError(
		f"Supabase {operation} failed: {type(exc).__name__}"
	)


class SupabaseDatabase:
	"""Store document and chunk metadata in Supabase PostgreSQL tables."""

	def __init__(
		self,
		client: Any | None = None,
		url: str | None = None,
		key: str | None = None,
	) -> None:
		self._client = client
		self.url = url if url is not None else settings.SUPABASE_URL
		self.key = key if key is not None else settings.SUPABASE_KEY

	@property
	def configured(self) -> bool:
		return bool(self.url and self.key)

	@property
	def client(self) -> Any:
		if self._client is None:
			if not self.url or not self.key:
				raise SupabaseDatabaseError(
					"SUPABASE_URL and SUPABASE_KEY must be configured."
				)
			try:
				from supabase import create_client

				self._client = create_client(self.url, self.key)
			except Exception as exc:
				raise _record_error("client initialization", exc) from exc
		return self._client

	def insert_document(
		self,
		document_id: str,
		filename: str,
		file_type: str,
		source: str,
		processing_status: str = "uploaded",
		upload_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Insert one document metadata record and return the stored record."""
		record = {
			"document_id": document_id,
			"filename": filename,
			"file_type": file_type,
			"source": source,
			"upload_date": (upload_date or datetime.now(timezone.utc)).isoformat(),
			"processing_status": processing_status,
		}
		try:
			response = self.client.table("documents").insert(record).execute()
			return _first_record(response, "document insert")
		except SupabaseDatabaseError:
			raise
		except Exception as exc:
			raise _record_error("document insert", exc) from exc

	def get_document(self, document_id: str) -> dict[str, Any] | None:
		"""Retrieve one document record by its stable ID."""
		try:
			response = (
				self.client.table("documents")
				.select("*")
				.eq("document_id", document_id)
				.execute()
			)
			records = _response_records(response)
			return records[0] if records else None
		except Exception as exc:
			raise _record_error("document lookup", exc) from exc

	def insert_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
		"""Insert chunk metadata records; vector data remains in FAISS."""
		records = [
			{
				"chunk_id": chunk["chunk_id"],
				"document_id": chunk["document_id"],
				"filename": chunk["filename"],
				"page_number": chunk.get("page_number"),
				"text": chunk["text"],
				"metadata": dict(chunk.get("metadata", {})),
			}
			for chunk in chunks
		]
		if not records:
			return []
		try:
			response = self.client.table("chunks").insert(records).execute()
			return _response_records(response)
		except Exception as exc:
			raise _record_error("chunk insert", exc) from exc

	def get_chunks(self, document_id: str) -> list[dict[str, Any]]:
		"""Retrieve all chunk metadata for one document only."""
		try:
			response = (
				self.client.table("chunks")
				.select("*")
				.eq("document_id", document_id)
				.execute()
			)
			return _response_records(response)
		except Exception as exc:
			raise _record_error("chunk lookup", exc) from exc


def _response_records(response: Any) -> list[dict[str, Any]]:
	data = getattr(response, "data", None)
	if data is None:
		raise ValueError("Supabase response did not contain data")
	if not isinstance(data, list):
		raise ValueError("Supabase response data must be a list")
	return data


def _first_record(response: Any, operation: str) -> dict[str, Any]:
	records = _response_records(response)
	if not records:
		raise SupabaseDatabaseError(f"Supabase {operation} returned no record")
	return records[0]


_database = SupabaseDatabase()


def get_database() -> SupabaseDatabase:
	"""Return the process-wide Supabase database access object."""
	return _database

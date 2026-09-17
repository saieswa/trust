-- Run this schema in the Supabase SQL editor.

create table if not exists public.documents (
    document_id text primary key,
    filename text not null,
    file_type text not null,
    source text not null,
    upload_date timestamptz not null default now(),
    processing_status text not null default 'uploaded'
);

create table if not exists public.chunks (
    chunk_id text primary key,
    document_id text not null references public.documents(document_id) on delete cascade,
    filename text not null,
    page_number text,
    text text not null,
    metadata jsonb not null default '{}'::jsonb
);

create index if not exists chunks_document_id_idx
    on public.chunks (document_id);
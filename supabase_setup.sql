-- FinxView workspace persistence — run this ONCE in your Supabase project's
-- SQL editor (Dashboard → SQL Editor → New query → paste this → Run).
--
-- This creates a single table that stores your workspace as one JSON blob:
-- drawings, alerts, favorite tools, drawing templates, and the favorites-bar
-- position. The app reads/writes it through your local server only — your
-- Supabase key never reaches the browser.

create table if not exists app_state (
  key        text primary key,
  value      jsonb not null,
  updated_at timestamptz not null default now()
);

-- Keep updated_at current on every save
create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists app_state_set_updated_at on app_state;
create trigger app_state_set_updated_at
  before update on app_state
  for each row
  execute function set_updated_at();

-- The app connects with the service_role key (kept only in your local
-- .env, never sent to the browser), which bypasses Row Level Security by
-- design — so no RLS policy is required for this table to work.

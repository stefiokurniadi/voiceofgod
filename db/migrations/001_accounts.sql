CREATE TABLE churches (
    id text PRIMARY KEY,
    name text NOT NULL,
    note text NOT NULL DEFAULT '',
    keep_names text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id text PRIMARY KEY,
    username text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    role text NOT NULL CHECK (role IN ('admin', 'operator')),
    disabled boolean NOT NULL DEFAULT false,
    church_id text REFERENCES churches (id),
    speaker_language text NOT NULL DEFAULT 'en' CHECK (speaker_language IN ('en', 'id')),
    tts_enabled boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE login_sessions (
    token_hash text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL
);

CREATE TABLE login_failures (
    username text NOT NULL,
    failed_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX login_failures_username_idx ON login_failures (username, failed_at);

CREATE TABLE services (
    id text PRIMARY KEY,
    church_id text NOT NULL REFERENCES churches (id),
    name text NOT NULL,
    note text NOT NULL DEFAULT '',
    service_date date,
    speaker_language text NOT NULL CHECK (speaker_language IN ('en', 'id')),
    voice boolean NOT NULL DEFAULT false,
    token_hash text NOT NULL,
    document jsonb NOT NULL,
    status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX services_church_idx ON services (church_id, created_at DESC);

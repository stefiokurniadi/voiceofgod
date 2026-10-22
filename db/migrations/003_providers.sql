ALTER TABLE users ADD COLUMN stt_provider text NOT NULL DEFAULT 'default' CHECK (stt_provider IN ('default', 'soniox'));
ALTER TABLE users ADD COLUMN tts_provider text NOT NULL DEFAULT 'google' CHECK (tts_provider IN ('google', 'soniox'));

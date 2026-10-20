ALTER TABLE users DROP CONSTRAINT IF EXISTS users_speaker_language_check;
ALTER TABLE users ADD CONSTRAINT users_speaker_language_check CHECK (speaker_language IN ('en', 'id', 'mix'));
ALTER TABLE users ADD COLUMN translation_language text NOT NULL DEFAULT 'id' CHECK (translation_language IN ('en', 'id'));

ALTER TABLE services DROP CONSTRAINT IF EXISTS services_speaker_language_check;
ALTER TABLE services ADD CONSTRAINT services_speaker_language_check CHECK (speaker_language IN ('en', 'id', 'mix'));
ALTER TABLE services ADD COLUMN translation_language text NOT NULL DEFAULT 'id' CHECK (translation_language IN ('en', 'id'));

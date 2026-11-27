-- Announcement intelligence: deterministic classification and transparent priority.
-- Apply after 002_indian_equity_intelligence.sql.

ALTER TABLE corporate_announcements
    ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'other',
    ADD COLUMN IF NOT EXISTS priority TEXT NOT NULL DEFAULT 'low',
    ADD COLUMN IF NOT EXISTS classification_method TEXT NOT NULL DEFAULT 'unclassified';

ALTER TABLE corporate_announcements
    DROP CONSTRAINT IF EXISTS ck_corporate_announcements_priority;
ALTER TABLE corporate_announcements
    ADD CONSTRAINT ck_corporate_announcements_priority
    CHECK (priority IN ('low', 'medium', 'high'));

CREATE INDEX IF NOT EXISTS ix_announcements_category_published
    ON corporate_announcements(category, published_at DESC);
CREATE INDEX IF NOT EXISTS ix_announcements_priority_published
    ON corporate_announcements(priority, published_at DESC);

-- 0005_two_phase_pipeline.sql
-- Split the monolithic pipeline into Phase 1 (rallies) and Phase 2 (analytics).

BEGIN;

-- 1. Drop the old CHECK constraint and re-add with new values.
ALTER TABLE videos DROP CONSTRAINT IF EXISTS videos_status_check;
ALTER TABLE videos ADD CONSTRAINT videos_status_check
  CHECK (status IN (
    'pending',
    'uploaded',
    'processing',           -- legacy, tolerated for historical rows
    'processing_phase1',
    'phase1_complete',
    'processing_phase2',
    'completed',
    'failed',               -- legacy, tolerated for historical rows
    'failed_phase1',
    'failed_phase2'
  ));

-- 2. Remap legacy 'failed' rows to 'failed_phase2' (closest semantic match).
UPDATE videos SET status='failed_phase2' WHERE status='failed';

-- 3. Add phase column to processing_logs.
ALTER TABLE processing_logs
  ADD COLUMN IF NOT EXISTS phase TEXT
    CHECK (phase IS NULL OR phase IN ('phase1', 'phase2'));

COMMIT;

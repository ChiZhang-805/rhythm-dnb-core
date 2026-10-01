"""Explicit SQLite stores. Storage never predicts, trains, imputes, or rewrites text."""

from contextlib import contextmanager
from datetime import date, datetime
import json
from pathlib import Path
import sqlite3
from ..contracts import DailyPanel, FeatureValue, Provenance
from ..provenance import canonical_json, fingerprint
from ..timebase import instant
from ..definitions import MEASUREMENT_ID


STORE_SQL = '''
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE observations(id TEXT PRIMARY KEY, participant TEXT NOT NULL,
    end_at TEXT NOT NULL, available_at TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TABLE panels(participant TEXT NOT NULL, day TEXT NOT NULL, measurement_id TEXT NOT NULL,
    payload TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY(participant,day,measurement_id));
CREATE TABLE outcomes(id TEXT PRIMARY KEY, participant TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE predictions(id TEXT PRIMARY KEY, participant TEXT NOT NULL,
    issued_at TEXT NOT NULL, bundle_id TEXT NOT NULL, payload TEXT NOT NULL,
    UNIQUE(participant,issued_at,bundle_id));
CREATE TABLE audit(id INTEGER PRIMARY KEY, operation TEXT NOT NULL, content_hash TEXT NOT NULL);
CREATE INDEX observations_asof ON observations(participant,available_at,end_at);
'''


class RhythmRepository:
    def __init__(self, path, *, readonly=True):
        # PSEUDOCODE: retain an explicit path; opening the object does not create a database.
        self.path = Path(path).resolve()
        self.readonly = readonly

    @contextmanager
    def connection(self, *, _initialize=False):
        # PSEUDOCODE: connect in requested mode -> enforce foreign keys/transaction -> always close.
        mode = 'ro' if self.readonly else 'rw'
        db = sqlite3.connect(self.path.as_uri() + '?mode=' + mode, uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if _initialize:
                if tables:
                    raise ValueError('Initialization requires an empty store.')
            elif 'metadata' not in tables or db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone() is None or db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0] != fingerprint(STORE_SQL):
                raise ValueError('Store structure differs from the active data contract; rebuild from source evidence into a separate store.')
            with db:
                yield db
        finally:
            db.close()

    @classmethod
    def create(cls, path):
        # PSEUDOCODE: reserve a new file exclusively, then initialize normalized append-only tables.
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb'):
            pass
        repo = cls(path, readonly=False)
        with repo.connection(_initialize=True) as db:
            db.executescript(STORE_SQL)
            db.execute('INSERT INTO metadata VALUES (?,?)', ('schema', fingerprint(STORE_SQL)))
        return repo

    def append_observations(self, rows):
        # PSEUDOCODE: validate all inputs before opening one atomic insertion transaction.
        from .validation import validate_observations
        rows = validate_observations(rows)
        with self.connection() as db:
            for row in rows:
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',
                           (row.observation_id, row.participant_id, instant(row.end).isoformat(),
                            instant(row.available_at).isoformat(), canonical_json(row), fingerprint(row)))
            db.execute('INSERT INTO audit(operation,content_hash) VALUES (?,?)', ('append_observations', fingerprint(rows)))

    def append_panel(self, panel):
        # PSEUDOCODE: insert an immutable panel tied to its measurement definition; never overwrite a previously used feature set.
        if any(f.measurement_id != MEASUREMENT_ID for f in panel.features):
            raise ValueError('Stored panel and active measurement definitions differ.')
        with self.connection() as db:
            db.execute('INSERT INTO panels VALUES (?,?,?,?,?)',
                       (panel.participant_id, panel.day.isoformat(), MEASUREMENT_ID, canonical_json(panel), fingerprint(panel)))
            db.execute('INSERT INTO audit(operation,content_hash) VALUES (?,?)', ('append_panel', fingerprint(panel)))

    def read_features_as_of(self, participant_id, as_of):
        # PSEUDOCODE: read only measurement panels -> remove unavailable features -> reconstruct contracts.
        with self.connection() as db:
            rows = db.execute('SELECT payload,hash FROM panels WHERE participant=? AND measurement_id=? ORDER BY day',
                              (participant_id, MEASUREMENT_ID)).fetchall()
        panels = []
        for row in rows:
            payload = json.loads(row['payload'])
            if fingerprint(payload) != row['hash']:
                raise ValueError('Stored panel checksum mismatch.')
            features = []
            for value in payload['features']:
                if instant(value['available_at']) <= instant(as_of):
                    value['available_at'] = datetime.fromisoformat(value['available_at'])
                    if value.get('measured_until'):
                        value['measured_until'] = datetime.fromisoformat(value['measured_until'])
                    provenance = value['provenance']; provenance['parent_ids'] = tuple(provenance['parent_ids'])
                    value['provenance'] = Provenance(**provenance)
                    features.append(FeatureValue(**value))
            panels.append(DailyPanel(payload['participant_id'], date.fromisoformat(payload['day']),
                                     payload['timezone'], tuple(features)))
        return tuple(panels)

    def append_prediction(self, prediction):
        # PSEUDOCODE: append one unique forecast and its policy state; duplicate issuance is rejected.
        with self.connection() as db:
            db.execute('INSERT INTO predictions VALUES (?,?,?,?,?)',
                       (fingerprint(prediction), prediction.participant_id, instant(prediction.issued_at).isoformat(),
                        prediction.bundle_id, canonical_json(prediction)))
            db.execute('INSERT INTO audit(operation,content_hash) VALUES (?,?)', ('append_prediction', fingerprint(prediction)))


class TextRepository:
    def __init__(self, path):
        # PSEUDOCODE: keep text corpus ownership separate from rhythm observation storage.
        self.path = Path(path).resolve()

    def read_legacy_examples(self):
        # PSEUDOCODE: open legacy corpus read-only and expose stored rows without changing their meaning.
        db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            table = 'chinese_examples' if 'chinese_examples' in tables else 'examples' if 'examples' in tables else 'records' if 'records' in tables else None
            if table is None:
                raise ValueError('No recognized text example table.')
            return [dict(r) for r in db.execute('SELECT * FROM ' + table)]
        finally:
            db.close()

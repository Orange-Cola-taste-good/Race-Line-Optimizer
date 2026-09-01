import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import TrackCanvas from './components/TrackCanvas';
import CurvatureChart from './components/CurvatureChart';
import DropZone from './components/DropZone';

const API = '';  // proxy to localhost:8000

const S = {
  app: {
    minHeight: '100vh',
    background: '#0d0d0f',
    color: '#e8e8ec',
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    padding: '20px 32px',
    borderBottom: '1px solid #1f1f27',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  title: { fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px' },
  accent: { color: '#e11d48' },
  body: {
    display: 'flex',
    flex: 1,
    gap: 0,
  },
  sidebar: {
    width: 300,
    minWidth: 260,
    borderRight: '1px solid #1f1f27',
    padding: 24,
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
    overflowY: 'auto',
  },
  main: {
    flex: 1,
    padding: 24,
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  label: { fontSize: 11, fontWeight: 600, letterSpacing: 1, color: '#666', textTransform: 'uppercase', marginBottom: 6 },
  tabRow: { display: 'flex', gap: 4 },
  tab: (active) => ({
    flex: 1,
    padding: '8px 4px',
    background: active ? '#e11d48' : '#1a1a22',
    color: active ? '#fff' : '#888',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 600,
    transition: 'all 0.15s',
  }),
  select: {
    width: '100%',
    background: '#1a1a22',
    color: '#e8e8ec',
    border: '1px solid #2a2a36',
    borderRadius: 6,
    padding: '8px 10px',
    fontSize: 14,
  },
  input: {
    width: '100%',
    background: '#1a1a22',
    color: '#e8e8ec',
    border: '1px solid #2a2a36',
    borderRadius: 6,
    padding: '8px 10px',
    fontSize: 14,
  },
  btn: (disabled) => ({
    width: '100%',
    padding: '10px',
    background: disabled ? '#2a1a20' : '#e11d48',
    color: disabled ? '#666' : '#fff',
    border: 'none',
    borderRadius: 6,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontWeight: 700,
    fontSize: 14,
    transition: 'background 0.15s',
  }),
  viewRow: { display: 'flex', gap: 4 },
  viewBtn: (active) => ({
    flex: 1,
    padding: '6px 4px',
    background: active ? '#1f1f27' : 'transparent',
    color: active ? '#e8e8ec' : '#555',
    border: '1px solid ' + (active ? '#333' : 'transparent'),
    borderRadius: 5,
    cursor: 'pointer',
    fontSize: 11,
    fontWeight: 600,
  }),
  stat: {
    background: '#1a1a22',
    borderRadius: 6,
    padding: '10px 14px',
    fontSize: 13,
  },
  statLabel: { color: '#666', fontSize: 11, marginBottom: 2 },
  statVal: { fontWeight: 700, fontSize: 16 },
  error: {
    background: '#2a0a10',
    border: '1px solid #7f1d1d',
    borderRadius: 6,
    padding: '10px 14px',
    fontSize: 13,
    color: '#fca5a5',
  },
  maskPreview: {
    borderRadius: 6,
    width: '100%',
    border: '1px solid #2a2a36',
    marginTop: 8,
  },
};

export default function App() {
  const [tracks, setTracks]       = useState([]);
  const [mode, setMode]           = useState('known');       // 'known' | 'schematic' | 'photo'
  const [selected, setSelected]   = useState('');
  const [trackData, setTrackData] = useState(null);
  const [result, setResult]       = useState(null);
  const [view, setView]           = useState('ml+physics');  // 'ml' | 'ml+physics' | 'ground_truth'
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [scaleKm, setScaleKm]     = useState('5.0');

  // Load track list on mount
  useEffect(() => {
    axios.get(`${API}/tracks`).then(r => {
      setTracks(r.data);
      if (r.data.length > 0) setSelected(r.data[0]);
    }).catch(() => setError('Cannot reach API at localhost:8000'));
  }, []);

  // Load track geometry when selection changes
  useEffect(() => {
    if (mode !== 'known' || !selected) return;
    axios.get(`${API}/tracks/${selected}`).then(r => {
      setTrackData({ centerline: r.data.centerline });
      setResult(null);
    });
  }, [selected, mode]);

  const clearState = () => { setResult(null); setError(null); setImageFile(null); setTrackData(null); };

  const handleModeChange = m => { setMode(m); clearState(); };

  const predict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (mode === 'known') {
        const r = await axios.post(`${API}/predict`, { name: selected });
        setResult(r.data);
      } else if (mode === 'schematic' || mode === 'photo') {
        if (!imageFile) { setError('Please drop an image first.'); return; }
        const form = new FormData();
        form.append('file', imageFile);
        form.append('scale_km', scaleKm);
        const endpoint = mode === 'schematic' ? '/predict/image' : '/predict/photo';
        const r = await axios.post(`${API}${endpoint}`, form, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setResult(r.data);
        if (!trackData) setTrackData({ centerline: r.data.raceline });
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, [mode, selected, imageFile, scaleKm, trackData]);

  const canPredict = !loading && (
    (mode === 'known' && selected) ||
    ((mode === 'schematic' || mode === 'photo') && imageFile)
  );

  const hasGroundTruth = result?.ground_truth;

  return (
    <div style={S.app}>
      {/* Header */}
      <header style={S.header}>
        <span style={{ fontSize: 22 }}>🏎</span>
        <span style={S.title}>Raceline <span style={S.accent}>Optimizer</span></span>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: '#444' }}>
          {tracks.length} tracks loaded
        </span>
      </header>

      <div style={S.body}>
        {/* Sidebar */}
        <aside style={S.sidebar}>
          {/* Mode tabs */}
          <div>
            <div style={S.label}>Input mode</div>
            <div style={S.tabRow}>
              {[
                { id: 'known',     label: 'Known' },
                { id: 'schematic', label: 'Schematic' },
                { id: 'photo',     label: 'Photo' },
              ].map(({ id, label }) => (
                <button key={id} style={S.tab(mode === id)} onClick={() => handleModeChange(id)}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Mode-specific inputs */}
          {mode === 'known' && (
            <div>
              <div style={S.label}>Track</div>
              <select
                style={S.select}
                value={selected}
                onChange={e => setSelected(e.target.value)}
              >
                {tracks.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          )}

          {(mode === 'schematic' || mode === 'photo') && (
            <>
              <div>
                <div style={S.label}>{mode === 'schematic' ? 'Track schematic PNG' : 'Aerial / satellite photo'}</div>
                <DropZone
                  onFile={f => { setImageFile(f); setResult(null); setTrackData(null); }}
                  label={mode === 'schematic'
                    ? 'Drop a top-down track schematic'
                    : 'Drop an aerial or satellite image'}
                />
                {imageFile && (
                  <div style={{ fontSize: 12, color: '#666', marginTop: 6 }}>
                    ✓ {imageFile.name}
                  </div>
                )}
              </div>
              <div>
                <div style={S.label}>Track length (km) — for scale</div>
                <input
                  type="number"
                  style={S.input}
                  value={scaleKm}
                  step="0.1"
                  min="0.5"
                  onChange={e => setScaleKm(e.target.value)}
                />
              </div>
            </>
          )}

          {/* Predict button */}
          <button style={S.btn(!canPredict)} disabled={!canPredict} onClick={predict}>
            {loading ? 'Computing…' : 'Predict Raceline'}
          </button>

          {/* Error */}
          {error && <div style={S.error}>{error}</div>}

          {/* Stats */}
          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={S.label}>Result</div>
              <div style={S.stat}>
                <div style={S.statLabel}>Method</div>
                <div style={S.statVal}>{result.method}</div>
              </div>
              {result.source && (
                <div style={S.stat}>
                  <div style={S.statLabel}>Source</div>
                  <div style={S.statVal}>{result.source}</div>
                </div>
              )}
              <div style={S.stat}>
                <div style={S.statLabel}>Points</div>
                <div style={S.statVal}>{result.raceline?.length ?? '—'}</div>
              </div>
              {result.sam_mask_preview && (
                <div>
                  <div style={S.label}>SAM mask preview</div>
                  <img
                    src={`data:image/png;base64,${result.sam_mask_preview}`}
                    alt="SAM mask"
                    style={S.maskPreview}
                  />
                </div>
              )}
            </div>
          )}
        </aside>

        {/* Main canvas area */}
        <main style={S.main}>
          {/* View toggle */}
          {result && (
            <div>
              <div style={S.label}>View</div>
              <div style={S.viewRow}>
                <button style={S.viewBtn(view === 'ml')} onClick={() => setView('ml')}>ML only</button>
                <button style={S.viewBtn(view === 'ml+physics')} onClick={() => setView('ml+physics')}>ML + Physics</button>
                {hasGroundTruth && (
                  <button style={S.viewBtn(view === 'ground_truth')} onClick={() => setView('ground_truth')}>Ground truth</button>
                )}
              </div>
            </div>
          )}

          {/* Track canvas */}
          <TrackCanvas trackData={trackData} result={result} view={view} />

          {/* Curvature chart */}
          {result?.curvature && <CurvatureChart data={result.curvature} />}

          {/* Empty state */}
          {!trackData && !result && (
            <div style={{ color: '#444', fontSize: 14, textAlign: 'center', marginTop: 80 }}>
              Select a track or drop an image, then click Predict Raceline.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

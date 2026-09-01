export default function CurvatureChart({ data }) {
  if (!data || data.length === 0) return null;

  const W = 600, H = 160, PX = 40, PY = 20;
  const dw = W - PX * 2, dh = H - PY * 2;

  const maxDist = Math.max(...data.map(d => d.dist_m));
  const kappas  = data.map(d => d.kappa);
  const minK = Math.min(...kappas), maxK = Math.max(...kappas);
  const rangeK = maxK - minK || 1;

  const tx = dist => PX + (dist / maxDist) * dw;
  const ty = k => PY + dh - ((k - minK) / rangeK) * dh;

  const path = data.map((d, i) =>
    `${i === 0 ? 'M' : 'L'}${tx(d.dist_m).toFixed(1)},${ty(d.kappa).toFixed(1)}`
  ).join(' ');

  // Zero line
  const zeroY = ty(0).toFixed(1);

  // Y-axis ticks
  const ticks = [minK, 0, maxK].filter(v => isFinite(v));

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>Curvature profile (rad/m)</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: '#0d0d0f', borderRadius: 6 }}>
        {/* Zero line */}
        <line x1={PX} y1={zeroY} x2={W - PX} y2={zeroY} stroke="#333" strokeWidth="1" strokeDasharray="4,4" />

        {/* Curvature path */}
        <path d={path} fill="none" stroke="#e11d48" strokeWidth="1.5" />

        {/* Y-axis ticks */}
        {ticks.map(v => (
          <g key={v}>
            <line x1={PX - 4} y1={ty(v)} x2={PX} y2={ty(v)} stroke="#555" strokeWidth="1" />
            <text x={PX - 6} y={ty(v) + 4} textAnchor="end" fill="#666" fontSize="10">
              {v.toFixed(3)}
            </text>
          </g>
        ))}

        {/* X-axis label */}
        <text x={W / 2} y={H - 2} textAnchor="middle" fill="#555" fontSize="10">
          Distance (m)
        </text>

        {/* X-axis start/end */}
        <text x={PX} y={H - 2} textAnchor="start" fill="#555" fontSize="10">0</text>
        <text x={W - PX} y={H - 2} textAnchor="end" fill="#555" fontSize="10">
          {maxDist.toFixed(0)}
        </text>
      </svg>
    </div>
  );
}

import { useEffect, useRef } from 'react';

const PAD = 40;

function project(pts, W, H) {
  if (!pts || pts.length === 0) return [];
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
  const scaleX = (W - PAD * 2) / rangeX, scaleY = (H - PAD * 2) / rangeY;
  const scale = Math.min(scaleX, scaleY);
  const offX = PAD + ((W - PAD * 2) - rangeX * scale) / 2;
  const offY = PAD + ((H - PAD * 2) - rangeY * scale) / 2;
  return pts.map(p => ({
    x: offX + (p.x - minX) * scale,
    y: offY + (p.y - minY) * scale,
  }));
}

function drawLoop(ctx, pts, color, width) {
  if (!pts || pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

export default function TrackCanvas({ trackData, result, view }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!trackData) return;

    // Build combined point set for projection bounds
    const all = [
      ...(trackData.centerline || []),
      ...(result?.raceline || []),
    ];
    const projected = pts => project(pts, W, H);
    const ref = projected(all);
    if (ref.length === 0) return;

    // Derive transform from combined bounds
    const xs = all.map(p => p.x), ys = all.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
    const scaleX = (W - PAD * 2) / rangeX, scaleY = (H - PAD * 2) / rangeY;
    const scale = Math.min(scaleX, scaleY);
    const offX = PAD + ((W - PAD * 2) - rangeX * scale) / 2;
    const offY = PAD + ((H - PAD * 2) - rangeY * scale) / 2;
    const tx = pts => pts.map(p => ({
      x: offX + (p.x - minX) * scale,
      y: offY + (p.y - minY) * scale,
    }));

    // Centerline (grey dashed)
    if (trackData.centerline) {
      ctx.setLineDash([4, 4]);
      drawLoop(ctx, tx(trackData.centerline), '#555', 1);
      ctx.setLineDash([]);
    }

    // Ground truth (white, if toggled)
    if (view === 'ground_truth' && result?.ground_truth) {
      drawLoop(ctx, tx(result.ground_truth), '#ffffff', 2);
    }

    // ML only
    if ((view === 'ml' || view === 'ml+physics') && result?.raceline_ml) {
      drawLoop(ctx, tx(result.raceline_ml), '#facc15', view === 'ml' ? 2.5 : 1.5);
    }

    // ML + Physics (final)
    if (view === 'ml+physics' && result?.raceline) {
      drawLoop(ctx, tx(result.raceline), '#e11d48', 2.5);
    }

    // Legend
    const legend = [];
    if (view === 'ground_truth') legend.push({ color: '#ffffff', label: 'Ground truth' });
    if (view === 'ml') legend.push({ color: '#facc15', label: 'ML prediction' });
    if (view === 'ml+physics') {
      legend.push({ color: '#facc15', label: 'ML only' });
      legend.push({ color: '#e11d48', label: 'ML + Physics' });
    }
    legend.push({ color: '#555', label: 'Centerline' });

    legend.forEach(({ color, label }, i) => {
      ctx.fillStyle = color;
      ctx.fillRect(12, 12 + i * 20, 16, 3);
      ctx.fillStyle = '#aaa';
      ctx.font = '12px system-ui';
      ctx.fillText(label, 34, 22 + i * 20);
    });
  }, [trackData, result, view]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={500}
      style={{ background: '#111', borderRadius: 8, display: 'block', width: '100%', height: 'auto' }}
    />
  );
}

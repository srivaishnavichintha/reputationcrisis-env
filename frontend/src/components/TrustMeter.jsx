import { motion } from "framer-motion";

function getTrustColor(trust) {
  if (trust > 0.65) return "#7cfc00";
  if (trust > 0.40) return "#ffd700";
  if (trust > 0.20) return "#ff8c00";
  return "#ff1a1a";
}

function getTrustLabel(trust) {
  if (trust > 0.75) return "STRONG";
  if (trust > 0.55) return "STABLE";
  if (trust > 0.35) return "FRAGILE";
  if (trust > 0.20) return "CRITICAL";
  return "COLLAPSED";
}

export default function TrustMeter({ trust = 0.5 }) {
  const pct = trust * 100;
  const color = getTrustColor(trust);
  const label = getTrustLabel(trust);

  // SVG arc parameters
  const R = 48;
  const cx = 64, cy = 72;
  const startAngle = -210;
  const sweepAngle = 240;
  const endAngle = startAngle + sweepAngle * trust;

  function polarToCartesian(cx, cy, r, angleDeg) {
    const rad = (angleDeg - 90) * Math.PI / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  }

  function arcPath(cx, cy, r, startDeg, endDeg) {
    const start = polarToCartesian(cx, cy, r, startDeg);
    const end = polarToCartesian(cx, cy, r, endDeg);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`;
  }

  const trackPath = arcPath(cx, cy, R, startAngle, startAngle + sweepAngle);
  const fillPath = arcPath(cx, cy, R, startAngle, endAngle);

  return (
    <div className="trust-meter">
      <svg viewBox="0 0 128 100" className="trust-svg">
        {/* Track */}
        <path
          d={trackPath}
          fill="none"
          stroke="#1a1a2e"
          strokeWidth={8}
          strokeLinecap="round"
        />
        {/* Animated fill */}
        <motion.path
          d={fillPath}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: trust }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
        {/* Glow effect on fill */}
        <motion.path
          d={fillPath}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinecap="round"
          opacity={0.4}
          animate={{ opacity: [0.4, 0.8, 0.4] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        {/* Center value */}
        <text x={cx} y={cy - 8} textAnchor="middle" fill="#fff" fontSize="18" fontWeight="bold" fontFamily="monospace">
          {pct.toFixed(0)}
        </text>
        <text x={cx} y={cy + 6} textAnchor="middle" fill="#555" fontSize="8" fontFamily="monospace">
          %
        </text>
      </svg>

      <div className="trust-label" style={{ color }}>
        {label}
      </div>

      {trust < 0.25 && (
        <motion.div
          className="trust-warning"
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 0.6, repeat: Infinity }}
        >
          ⚠ TRUST COLLAPSE IMMINENT
        </motion.div>
      )}
    </div>
  );
}

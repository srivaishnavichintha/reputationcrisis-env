import { motion } from "framer-motion";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Dot
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-step">Step {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {(p.value * (p.name === "sentiment" ? 100 : 100)).toFixed(1)}%
        </p>
      ))}
    </div>
  );
};

const ActionDot = (props) => {
  const { cx, cy, payload } = props;
  if (!payload?.action) return null;
  return (
    <Dot cx={cx} cy={cy} r={4} fill="#fff" stroke="#00d4ff" strokeWidth={2} />
  );
};

export default function SentimentChart({ history = [] }) {
  const data = history.map(h => ({
    step: h.step,
    sentiment: parseFloat(((h.sentiment + 1) / 2 * 100).toFixed(1)), // 0-100
    trust: parseFloat((h.trust * 100).toFixed(1)),
    virality: parseFloat((h.virality * 100).toFixed(1)),
    action: h.action,
  }));

  return (
    <motion.div
      className="chart-wrapper"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {data.length < 1 ? (
        <div className="chart-empty">
          <span className="chart-spinner">◈</span> Connecting to backend...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="trustGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#7cfc00" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#7cfc00" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="viralGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ff4444" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#ff4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="#ffffff08" />
            <XAxis
              dataKey="step"
              tick={{ fill: "#555", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#555", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine y={50} stroke="#ffffff15" strokeDasharray="3 3" />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="virality"
              stroke="#ff4444"
              strokeWidth={1.5}
              fill="url(#viralGrad)"
              dot={false}
              name="virality"
            />
            <Area
              type="monotone"
              dataKey="trust"
              stroke="#7cfc00"
              strokeWidth={1.5}
              fill="url(#trustGrad)"
              dot={false}
              name="trust"
            />
            <Area
              type="monotone"
              dataKey="sentiment"
              stroke="#00d4ff"
              strokeWidth={2}
              fill="url(#sentGrad)"
              dot={<ActionDot />}
              name="sentiment"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
      <div className="chart-legend">
        <span className="legend-item"><span className="dot" style={{ background: "#00d4ff" }} />Sentiment</span>
        <span className="legend-item"><span className="dot" style={{ background: "#7cfc00" }} />Trust</span>
        <span className="legend-item"><span className="dot" style={{ background: "#ff4444" }} />Virality</span>
      </div>
    </motion.div>
  );
}

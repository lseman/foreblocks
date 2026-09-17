type KpiCardProps = {
  label: string;
  value: string | number;
  tone?: 'gold' | 'cyan' | 'mint';
};

export function KpiCard({ label, value, tone = 'gold' }: KpiCardProps) {
  return (
    <article className={`kpi-card tone-${tone}`}>
      <p className="kpi-label">{label}</p>
      <h3 className="kpi-value">{value}</h3>
    </article>
  );
}

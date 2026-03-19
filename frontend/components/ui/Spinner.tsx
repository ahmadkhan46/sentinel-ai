export function Spinner({ size = 6 }: { size?: number }) {
  const px = size * 4;
  return (
    <div
      style={{ width: px, height: px }}
      className="animate-spin rounded-full border-2 border-slate-200 border-t-violet-600"
    />
  );
}

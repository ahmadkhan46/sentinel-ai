import { clsx } from "clsx";

export function Card({
  children,
  className,
  glass,
}: {
  children: React.ReactNode;
  className?: string;
  glass?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-2xl border",
        glass
          ? "border-white/20 bg-white/70 shadow-xl backdrop-blur-md"
          : "border-white/60 bg-white shadow-sm shadow-slate-200/60",
        className
      )}
    >
      {children}
    </div>
  );
}

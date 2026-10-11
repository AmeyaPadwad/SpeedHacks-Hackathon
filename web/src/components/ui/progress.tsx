// shadcn/ui Progress (no Radix dependency; same markup and classes)
import { cn } from "@/lib/utils";

export function Progress({ value, className, indicator }: { value: number; className?: string; indicator?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div role="progressbar" aria-valuenow={v} className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-secondary", className)}>
      <div
        className={cn("h-full w-full flex-1 bg-primary transition-transform duration-500", indicator)}
        style={{ transform: `translateX(-${100 - v}%)` }}
      />
    </div>
  );
}

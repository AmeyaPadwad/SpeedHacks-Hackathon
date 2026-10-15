// Aceternity UI "Moving Border" (compact): a light dot that travels around a pill's border.
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function MovingBorder({ children, className, active = true, color = "#8b5cf6" }:
  { children: React.ReactNode; className?: string; active?: boolean; color?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-full p-[1.5px]", className)}>
      {active && (
        <motion.div
          className="absolute inset-[-100%]"
          style={{ background: `conic-gradient(from 0deg, transparent 0 70%, ${color} 85%, transparent 100%)` }}
          animate={{ rotate: 360 }}
          transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
        />
      )}
      <div className={cn("relative rounded-full bg-background", !active && "ring-1 ring-border")}>{children}</div>
    </div>
  );
}

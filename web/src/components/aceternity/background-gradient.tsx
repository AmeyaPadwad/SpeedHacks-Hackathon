// Aceternity UI "Background Gradient": an animated multi-colour glow behind a card.
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function BackgroundGradient({
  children, className, containerClassName, animate = true,
}: { children?: React.ReactNode; className?: string; containerClassName?: string; animate?: boolean }) {
  const variants = { initial: { backgroundPosition: "0 50%" }, animate: { backgroundPosition: ["0, 50%", "100% 50%", "0 50%"] } };
  const bg =
    "bg-[radial-gradient(circle_farthest-side_at_0_100%,#8b5cf6,transparent),radial-gradient(circle_farthest-side_at_100%_0,#3b82f6,transparent),radial-gradient(circle_farthest-side_at_100%_100%,#f59e0b,transparent),radial-gradient(circle_farthest-side_at_0_0,#ec4899,#141316)]";
  const motionProps = animate
    ? { variants, initial: "initial", animate: "animate", transition: { duration: 5, repeat: Infinity, repeatType: "reverse" as const } }
    : {};
  return (
    <div className={cn("group relative p-[2px]", containerClassName)}>
      <motion.div {...motionProps} style={{ backgroundSize: animate ? "400% 400%" : undefined }}
        className={cn("absolute inset-0 z-[1] rounded-2xl opacity-50 blur-lg transition duration-500 will-change-transform group-hover:opacity-80", bg)} />
      <motion.div {...motionProps} style={{ backgroundSize: animate ? "400% 400%" : undefined }}
        className={cn("absolute inset-0 z-[1] rounded-2xl will-change-transform", bg)} />
      <div className={cn("relative z-10", className)}>{children}</div>
    </div>
  );
}

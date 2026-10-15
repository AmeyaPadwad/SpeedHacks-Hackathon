/** A short notice at the top of the page (the desktop tray balloon). Plain text only. */
import { AnimatePresence, motion } from "framer-motion";
import type { Notice as NoticeT } from "../controller";

export function Notice({ notice }: { notice: NoticeT | null }) {
  return (
    <AnimatePresence>
      {notice && (
        <motion.div key={notice.id} role="status" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          className="fixed left-1/2 top-4 z-[70] w-[min(360px,calc(100vw-32px))] -translate-x-1/2 rounded-xl border bg-white/95 px-4 py-2.5 shadow-lg backdrop-blur">
          <div className="text-xs font-semibold">{notice.title}</div>
          <div className="text-[13px] text-zinc-700">{notice.text}</div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

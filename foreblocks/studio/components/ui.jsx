/**
 * shadcn-inspired UI primitives built on the studio CSS variable system.
 * Uses lucide-react for icons. No Radix UI dependency needed.
 */
import { clsx } from "clsx";

// ─── Badge ───────────────────────────────────────────────────────────────────

const badgeVariants = {
    default: "inline-flex items-center gap-1 rounded-md border border-[var(--border-strong)] bg-[var(--accent-soft)] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.07em] text-[var(--accent-strong)]",
    secondary: "inline-flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--secondary-soft)] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.07em] text-[var(--secondary)]",
    warm: "inline-flex items-center gap-1 rounded-md border border-[var(--warm-soft)] bg-[var(--warm-soft)] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.07em] text-[var(--warm)]",
    cool: "inline-flex items-center gap-1 rounded-md border border-[var(--cool-soft)] bg-[var(--cool-soft)] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.07em] text-[var(--cool)]",
    outline: "inline-flex items-center gap-1 rounded-md border border-[var(--border)] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.07em] text-[var(--subtext)]",
};

export function Badge({ variant = "default", className, children, ...props }) {
    return (
        <span className={clsx(badgeVariants[variant] ?? badgeVariants.default, className)} {...props}>
            {children}
        </span>
    );
}

// ─── Separator ───────────────────────────────────────────────────────────────

export function Separator({ className, orientation = "horizontal", ...props }) {
    return (
        <div
            role="separator"
            className={clsx(
                "shrink-0 bg-[var(--border)]",
                orientation === "horizontal" ? "h-px w-full" : "h-full w-px",
                className
            )}
            {...props}
        />
    );
}

// ─── Skeleton ────────────────────────────────────────────────────────────────

export function Skeleton({ className, ...props }) {
    return (
        <div
            className={clsx(
                "animate-pulse rounded-md bg-[var(--border)]",
                className
            )}
            {...props}
        />
    );
}

// ─── Card ─────────────────────────────────────────────────────────────────────

export function Card({ className, children, ...props }) {
    return (
        <div
            className={clsx(
                "rounded-[10px] border border-[var(--border)] bg-[rgba(255,255,255,0.42)] backdrop-blur-md",
                "dark:bg-[rgba(255,255,255,0.04)]",
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
}

export function CardHeader({ className, children, ...props }) {
    return (
        <div className={clsx("flex flex-col gap-1 p-4 pb-0", className)} {...props}>
            {children}
        </div>
    );
}

export function CardTitle({ className, children, ...props }) {
    return (
        <h3
            className={clsx("text-[15px] font-semibold leading-none tracking-tight text-[var(--text)]", className)}
            {...props}
        >
            {children}
        </h3>
    );
}

export function CardDescription({ className, children, ...props }) {
    return (
        <p className={clsx("text-[12px] text-[var(--subtext)]", className)} {...props}>
            {children}
        </p>
    );
}

export function CardContent({ className, children, ...props }) {
    return (
        <div className={clsx("p-4", className)} {...props}>
            {children}
        </div>
    );
}

export function CardFooter({ className, children, ...props }) {
    return (
        <div className={clsx("flex items-center p-4 pt-0", className)} {...props}>
            {children}
        </div>
    );
}

// ─── Alert ────────────────────────────────────────────────────────────────────

const alertVariants = {
    default: "border-[var(--border)] bg-[rgba(255,255,255,0.4)] dark:bg-[rgba(255,255,255,0.04)]",
    accent: "border-l-2 border-[var(--border)] border-l-[var(--accent)] bg-[var(--accent-soft)]",
    warm: "border-l-2 border-[var(--border)] border-l-[var(--warm)] bg-[var(--warm-soft)]",
    cool: "border-l-2 border-[var(--border)] border-l-[var(--cool)] bg-[var(--cool-soft)]",
    destructive: "border-l-2 border-[var(--border)] border-l-red-500 bg-red-500/10",
};

export function Alert({ variant = "default", className, children, ...props }) {
    return (
        <div
            role="alert"
            className={clsx(
                "relative rounded-[10px] border p-3.5",
                alertVariants[variant] ?? alertVariants.default,
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
}

export function AlertTitle({ className, children, ...props }) {
    return (
        <h5
            className={clsx(
                "mb-1 text-[10px] font-bold uppercase tracking-[0.08em] text-[var(--accent-strong)]",
                className
            )}
            {...props}
        >
            {children}
        </h5>
    );
}

export function AlertDescription({ className, children, ...props }) {
    return (
        <div className={clsx("text-[12px] leading-[1.5] text-[var(--subtext)]", className)} {...props}>
            {children}
        </div>
    );
}

// ─── Progress ─────────────────────────────────────────────────────────────────

export function Progress({ value = 0, className, ...props }) {
    return (
        <div
            className={clsx("relative h-1.5 w-full overflow-hidden rounded-full bg-[var(--border)]", className)}
            {...props}
        >
            <div
                className="h-full rounded-full bg-[var(--accent)] transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
            />
        </div>
    );
}

// ─── Tooltip (simple hover title wrapper) ─────────────────────────────────────

export function Tooltip({ children, title, className }) {
    return (
        <span className={clsx("relative group", className)} title={title}>
            {children}
        </span>
    );
}

// ─── Input ────────────────────────────────────────────────────────────────────

export function Input({ className, ...props }) {
    return (
        <input
            className={clsx(
                "flex h-9 w-full rounded-[8px] border border-[var(--border)] bg-white/90",
                "dark:bg-white/7 dark:border-white/10",
                "px-3 py-2 text-[13px] text-[var(--text)] outline-none",
                "placeholder:text-[var(--muted)]",
                "focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]",
                "disabled:cursor-not-allowed disabled:opacity-45",
                "transition-all duration-150",
                className
            )}
            {...props}
        />
    );
}

// ─── Label ───────────────────────────────────────────────────────────────────

export function Label({ className, children, ...props }) {
    return (
        <label
            className={clsx(
                "text-[10px] font-bold uppercase tracking-[0.08em] text-[var(--accent-strong)] opacity-85",
                className
            )}
            {...props}
        >
            {children}
        </label>
    );
}

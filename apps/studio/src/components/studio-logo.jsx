export function StudioLogo() {
    return (
        <svg
            viewBox="0 0 96 96"
            className="studio-logo"
            role="img"
            aria-label="foreBlocks Studio logo"
        >
            <defs>
                <linearGradient id="studio-logo-bg" x1="10" y1="8" x2="86" y2="88" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="var(--accent)" stopOpacity="0.18" />
                    <stop offset="0.55" stopColor="var(--secondary)" stopOpacity="0.12" />
                    <stop offset="1" stopColor="var(--warm)" stopOpacity="0.18" />
                </linearGradient>
                <linearGradient id="studio-logo-stroke" x1="24" y1="18" x2="74" y2="78" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="var(--accent)" />
                    <stop offset="0.6" stopColor="var(--secondary)" />
                    <stop offset="1" stopColor="var(--warm)" />
                </linearGradient>
                <linearGradient id="studio-logo-curve" x1="18" y1="62" x2="78" y2="28" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="var(--accent)" />
                    <stop offset="0.5" stopColor="var(--secondary)" />
                    <stop offset="1" stopColor="var(--warm)" />
                </linearGradient>
            </defs>

            <rect x="8" y="8" width="80" height="80" rx="24" fill="url(#studio-logo-bg)" />
            <rect x="18" y="20" width="46" height="14" rx="7" fill="none" stroke="url(#studio-logo-stroke)" strokeWidth="6" />
            <rect x="18" y="41" width="60" height="14" rx="7" fill="none" stroke="url(#studio-logo-stroke)" strokeWidth="6" opacity="0.92" />
            <rect x="18" y="62" width="34" height="14" rx="7" fill="none" stroke="url(#studio-logo-stroke)" strokeWidth="6" opacity="0.84" />
            <path
                d="M22 62 C34 55, 43 58, 50 49 S64 33, 74 34"
                fill="none"
                stroke="url(#studio-logo-curve)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
            <circle cx="74" cy="34" r="6" fill="var(--warm)" />
            <circle cx="22" cy="62" r="4" fill="var(--accent)" opacity="0.9" />
        </svg>
    );
}

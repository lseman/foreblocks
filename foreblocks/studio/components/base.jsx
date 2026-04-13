import { useEffect, useRef, useState } from "react";

const SVG_NS = "http://www.w3.org/2000/svg";
const XHTML_NS = "http://www.w3.org/1999/xhtml";

function sanitizeFigureName(value) {
    return String(value || "figure")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "") || "figure";
}

function downloadTextFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    downloadBlob(filename, blob);
}

function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function inlineSvgStyles(sourceNode, targetNode) {
    if (!(sourceNode instanceof Element) || !(targetNode instanceof Element)) {
        return;
    }

    const computed = window.getComputedStyle(sourceNode);
    const styleFragments = [
        ["fill", computed.fill],
        ["stroke", computed.stroke],
        ["stroke-width", computed.strokeWidth],
        ["stroke-dasharray", computed.strokeDasharray],
        ["stroke-linecap", computed.strokeLinecap],
        ["stroke-linejoin", computed.strokeLinejoin],
        ["opacity", computed.opacity],
        ["color", computed.color],
        ["font-family", computed.fontFamily],
        ["font-size", computed.fontSize],
        ["font-weight", computed.fontWeight],
        ["letter-spacing", computed.letterSpacing],
        ["text-transform", computed.textTransform],
        ["paint-order", computed.paintOrder],
        ["vector-effect", computed.vectorEffect],
    ].filter(([, value]) => value && value !== "normal" && value !== "none" && value !== "rgba(0, 0, 0, 0)");

    if (styleFragments.length > 0) {
        const existing = targetNode.getAttribute("style");
        const nextStyle = styleFragments.map(([key, value]) => `${key}: ${value};`).join(" ");
        targetNode.setAttribute("style", existing ? `${existing} ${nextStyle}` : nextStyle);
    }

    const sourceChildren = Array.from(sourceNode.children);
    const targetChildren = Array.from(targetNode.children);
    sourceChildren.forEach((child, index) => {
        inlineSvgStyles(child, targetChildren[index]);
    });
}

function inlineHtmlStyles(sourceNode, targetNode) {
    if (!(sourceNode instanceof Element) || !(targetNode instanceof Element)) {
        return;
    }

    const computed = window.getComputedStyle(sourceNode);
    targetNode.setAttribute(
        "style",
        [
            `box-sizing: ${computed.boxSizing};`,
            `display: ${computed.display};`,
            `position: ${computed.position};`,
            `width: ${computed.width};`,
            `height: ${computed.height};`,
            `margin: ${computed.margin};`,
            `padding: ${computed.padding};`,
            `border: ${computed.border};`,
            `border-radius: ${computed.borderRadius};`,
            `background: ${computed.background};`,
            `background-color: ${computed.backgroundColor};`,
            `color: ${computed.color};`,
            `font-family: ${computed.fontFamily};`,
            `font-size: ${computed.fontSize};`,
            `font-weight: ${computed.fontWeight};`,
            `letter-spacing: ${computed.letterSpacing};`,
            `line-height: ${computed.lineHeight};`,
            `text-transform: ${computed.textTransform};`,
            `text-align: ${computed.textAlign};`,
            `gap: ${computed.gap};`,
            `grid-template-columns: ${computed.gridTemplateColumns};`,
            `grid-template-rows: ${computed.gridTemplateRows};`,
            `align-items: ${computed.alignItems};`,
            `justify-items: ${computed.justifyItems};`,
            `justify-content: ${computed.justifyContent};`,
            `box-shadow: ${computed.boxShadow};`,
            `overflow: ${computed.overflow};`,
            `white-space: ${computed.whiteSpace};`,
        ].join(" "),
    );

    if (targetNode instanceof HTMLButtonElement) {
        targetNode.disabled = true;
    }

    const sourceChildren = Array.from(sourceNode.children);
    const targetChildren = Array.from(targetNode.children);
    sourceChildren.forEach((child, index) => {
        inlineHtmlStyles(child, targetChildren[index]);
    });
}

function buildSvgPayloadFromLiveSvg(shell, title) {
    const liveSvg = shell.querySelector("svg");
    if (!liveSvg) {
        return null;
    }

    const clone = liveSvg.cloneNode(true);
    inlineSvgStyles(liveSvg, clone);

    const bounds = liveSvg.getBoundingClientRect();
    const width = Math.max(1, Math.ceil(bounds.width));
    const height = Math.max(1, Math.ceil(bounds.height));

    clone.setAttribute("xmlns", SVG_NS);
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    if (!clone.getAttribute("viewBox")) {
        clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
    }

    if (title) {
        const titleNode = document.createElementNS(SVG_NS, "title");
        titleNode.textContent = title;
        clone.insertBefore(titleNode, clone.firstChild);
    }

    return {
        width,
        height,
        markup: new XMLSerializer().serializeToString(clone),
    };
}

function buildSvgPayloadFromHtmlShell(shell, title) {
    const clone = shell.cloneNode(true);
    clone.querySelectorAll(".figure-export-controls").forEach((node) => node.remove());
    clone.setAttribute("xmlns", XHTML_NS);
    inlineHtmlStyles(shell, clone);

    const bounds = shell.getBoundingClientRect();
    const width = Math.max(1, Math.ceil(bounds.width));
    const height = Math.max(1, Math.ceil(bounds.height));
    const wrapperTitle = title ? `<title>${title.replace(/[<&>]/g, "")}</title>` : "";
    const markup = [
        `<svg xmlns="${SVG_NS}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
        wrapperTitle,
        `<foreignObject width="100%" height="100%">${clone.outerHTML}</foreignObject>`,
        "</svg>",
    ].join("");

    return { width, height, markup };
}

function buildFigureExportPayload(shell, title) {
    return buildSvgPayloadFromLiveSvg(shell, title) ?? buildSvgPayloadFromHtmlShell(shell, title);
}

function exportSvgFromShell(shell, fileStem, title) {
    const payload = buildFigureExportPayload(shell, title);
    if (!payload) {
        return false;
    }

    downloadTextFile(`${fileStem}.svg`, payload.markup, "image/svg+xml;charset=utf-8");
    return true;
}

function renderSvgPayloadToCanvas(payload) {
    return new Promise((resolve, reject) => {
        const blob = new Blob([payload.markup], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const image = new Image();

        image.onload = () => {
            const scale = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
            const canvas = document.createElement("canvas");
            canvas.width = Math.max(1, Math.ceil(payload.width * scale));
            canvas.height = Math.max(1, Math.ceil(payload.height * scale));

            const context = canvas.getContext("2d");
            if (!context) {
                URL.revokeObjectURL(url);
                reject(new Error("Canvas 2D context is unavailable."));
                return;
            }

            context.scale(scale, scale);
            context.fillStyle = "#ffffff";
            context.fillRect(0, 0, payload.width, payload.height);
            context.drawImage(image, 0, 0, payload.width, payload.height);
            URL.revokeObjectURL(url);
            resolve(canvas);
        };

        image.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error("Could not render exported SVG for PDF output."));
        };

        image.src = url;
    });
}

function base64ToUint8Array(base64) {
    const binary = window.atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
}

function encodePdfObject(objectNumber, content) {
    return new TextEncoder().encode(`${objectNumber} 0 obj\n${content}\nendobj\n`);
}

function concatenateByteArrays(parts) {
    const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    parts.forEach((part) => {
        combined.set(part, offset);
        offset += part.length;
    });
    return combined;
}

function buildPdfBytes(jpegBytes, imageWidth, imageHeight) {
    // Crop the PDF page to the exported figure instead of fitting onto a
    // fixed letter-size sheet, which adds large empty margins around charts.
    const pageWidth = Number(Math.max(1, imageWidth).toFixed(2));
    const pageHeight = Number(Math.max(1, imageHeight).toFixed(2));
    const drawWidth = pageWidth;
    const drawHeight = pageHeight;
    const offsetX = 0;
    const offsetY = 0;
    const contentStream = `q
${drawWidth} 0 0 ${drawHeight} ${offsetX} ${offsetY} cm
/Im0 Do
Q
`;
    const contentBytes = new TextEncoder().encode(contentStream);
    const imageHeader = new TextEncoder().encode(
        `4 0 obj
<< /Type /XObject /Subtype /Image /Width ${Math.max(1, Math.round(imageWidth))} /Height ${Math.max(1, Math.round(imageHeight))} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>
stream
`,
    );
    const imageFooter = new TextEncoder().encode("\nendstream\nendobj\n");
    const objects = [
        encodePdfObject(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        encodePdfObject(2, "<< /Type /Pages /Count 1 /Kids [3 0 R] >>"),
        encodePdfObject(
            3,
            `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources << /ProcSet [/PDF /ImageC] /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>`,
        ),
        concatenateByteArrays([imageHeader, jpegBytes, imageFooter]),
        concatenateByteArrays([
            new TextEncoder().encode(`5 0 obj\n<< /Length ${contentBytes.length} >>\nstream\n`),
            contentBytes,
            new TextEncoder().encode("endstream\nendobj\n"),
        ]),
    ];

    const header = new TextEncoder().encode("%PDF-1.4\n");
    let offset = header.length;
    const offsets = [0];
    objects.forEach((objectBytes) => {
        offsets.push(offset);
        offset += objectBytes.length;
    });

    const xrefOffset = offset;
    const xrefEntries = offsets
        .map((entryOffset, index) => (
            index === 0
                ? "0000000000 65535 f \n"
                : `${String(entryOffset).padStart(10, "0")} 00000 n \n`
        ))
        .join("");
    const xref = new TextEncoder().encode(`xref
0 ${offsets.length}
${xrefEntries}trailer
<< /Size ${offsets.length} /Root 1 0 R >>
startxref
${xrefOffset}
%%EOF`);

    return concatenateByteArrays([header, ...objects, xref]);
}

async function exportPdfFromShell(shell, fileStem, title) {
    const payload = buildFigureExportPayload(shell, title);
    if (!payload) {
        return false;
    }

    const canvas = await renderSvgPayloadToCanvas(payload);
    const jpegDataUrl = canvas.toDataURL("image/jpeg", 0.95);
    const jpegBytes = base64ToUint8Array(jpegDataUrl.split(",")[1] ?? "");
    const pdfBytes = buildPdfBytes(jpegBytes, payload.width, payload.height);
    downloadBlob(`${fileStem}.pdf`, new Blob([pdfBytes], { type: "application/pdf" }));
    return true;
}

function buildFigureTitle(shell, panelTitle) {
    const chartTitle = shell.querySelector(".mini-chart-title")?.textContent?.trim();
    return [panelTitle, chartTitle].filter(Boolean).join(" - ") || panelTitle || "figure";
}

async function exportFigureShell(shell, panelTitle, format = "svg") {
    const figureTitle = buildFigureTitle(shell, panelTitle);
    const fileStem = sanitizeFigureName(figureTitle);

    if (format === "pdf") {
        await exportPdfFromShell(shell, fileStem, figureTitle);
        return;
    }

    exportSvgFromShell(shell, fileStem, figureTitle);
}

export function Panel({ title, kicker, children, actions, accent = false }) {
    const panelRef = useRef(null);

    useEffect(() => {
        const panelNode = panelRef.current;
        if (!panelNode) {
            return undefined;
        }

        const figureShells = Array.from(panelNode.querySelectorAll(".chart-shell"));
        figureShells.forEach((shell) => {
            const hasControls = Array.from(shell.children).some((child) => child.classList?.contains("figure-export-controls"));
            if (hasControls) {
                return;
            }

            const controls = document.createElement("div");
            controls.className = "figure-export-controls";

            [
                {
                    label: "SVG",
                    format: "svg",
                    titleText: "Download figure as SVG",
                },
                {
                    label: "PDF",
                    format: "pdf",
                    titleText: "Download figure as PDF",
                },
            ].forEach(({ label, format, titleText }) => {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "figure-export-button";
                button.textContent = label;
                button.title = titleText;
                button.setAttribute("aria-label", `Download ${buildFigureTitle(shell, title)} as ${label}`);
                button.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    void exportFigureShell(shell, title, format).catch((error) => {
                        console.error(`Failed to export ${label} figure`, error);
                    });
                });
                controls.appendChild(button);
            });

            shell.appendChild(controls);
        });

        return () => {
            panelNode.querySelectorAll(".figure-export-controls").forEach((controls) => {
                controls.replaceWith();
            });
        };
    });

    return (
        <section ref={panelRef} className={`panel ${accent ? "panel-accent" : ""}`}>
            <div className="panel-head">
                <div>
                    {kicker ? <div className="panel-kicker">{kicker}</div> : null}
                    <h3 className="panel-title">{title}</h3>
                </div>
                {actions ? <div className="panel-actions">{actions}</div> : null}
            </div>
            {children}
        </section>
    );
}

export function HelpButton({ title, content }) {
    const [open, setOpen] = useState(false);

    useEffect(() => {
        if (!open) {
            return undefined;
        }

        const typesetMath = async () => {
            if (typeof window !== "undefined" && window.MathJax?.typesetPromise) {
                try {
                    await window.MathJax.typesetPromise();
                } catch {
                    // ignore math rendering failures
                }
            }
        };

        typesetMath();
        return undefined;
    }, [open, title, content]);

    return (
        <>
            <button
                type="button"
                className="ghost-button"
                title={`Open help for ${title}`}
                aria-label={`Open help for ${title}`}
                onClick={() => setOpen(true)}
            >
                ?
            </button>
            {open ? (
                <div className="modal-overlay help-overlay" role="presentation" onClick={() => setOpen(false)}>
                    <div
                        className="modal-dialog modal-code-dialog help-dialog"
                        role="dialog"
                        aria-modal="true"
                        aria-label={title}
                        onClick={(event) => event.stopPropagation()}
                    >
                        <button
                            className="modal-close"
                            type="button"
                            aria-label={`Close ${title} help`}
                            onClick={() => setOpen(false)}
                        >
                            Close
                        </button>
                        <Panel title={title} kicker="Algorithm help">
                            <div className="help-copy">
                                {Array.isArray(content)
                                    ? content.map((html, index) => (
                                        <div key={index} dangerouslySetInnerHTML={{ __html: html }} />
                                    ))
                                    : <div dangerouslySetInnerHTML={{ __html: content }} />}
                            </div>
                        </Panel>
                    </div>
                </div>
            ) : null}
        </>
    );
}

export function StepButton({ active, index, label, onClick }) {
    return (
        <button className={`step-button ${active ? "step-button-active" : ""}`} onClick={onClick}>
            <span className="step-index">{active ? "●" : index + 1}</span>
            <span>{label}</span>
        </button>
    );
}

function InputField({ label, hint, children }) {
    return (
        <label className="field">
            <span className="field-label">{label}</span>
            {children}
            {hint ? <span className="field-hint">{hint}</span> : null}
        </label>
    );
}

export function TextField({ label, value, onChange, hint, placeholder }) {
    return (
        <InputField label={label} hint={hint}>
            <input
                className="field-input"
                type="text"
                value={value}
                placeholder={placeholder}
                onChange={(event) => onChange(event.target.value)}
            />
        </InputField>
    );
}

export function NumberField({ label, value, onChange, min, max, step = 1, hint }) {
    return (
        <InputField label={label} hint={hint}>
            <input
                className="field-input"
                type="number"
                value={value}
                min={min}
                max={max}
                step={step}
                onChange={(event) => onChange(Number(event.target.value))}
            />
        </InputField>
    );
}

export function SelectField({ label, value, onChange, options, hint }) {
    return (
        <InputField label={label} hint={hint}>
            <select className="field-input" value={value} onChange={(event) => onChange(event.target.value)}>
                {options.map((option) => {
                    if (typeof option === "string") {
                        return (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        );
                    }
                    return (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    );
                })}
            </select>
        </InputField>
    );
}

export function ToggleField({ label, checked, onChange, hint }) {
    return (
        <label className="toggle-row">
            <div>
                <div className="field-label">{label}</div>
                {hint ? <div className="field-hint">{hint}</div> : null}
            </div>
            <button type="button" className={`toggle ${checked ? "toggle-on" : ""}`} onClick={() => onChange(!checked)}>
                <span className="toggle-knob" />
            </button>
        </label>
    );
}

export function StatPill({ label, value, tone = "accent" }) {
    return (
        <div className={`stat-pill stat-pill-${tone}`}>
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    );
}

export function PipelineNode({ title, detail, tone }) {
    return (
        <div className={`pipeline-node pipeline-node-${tone}`}>
            <div className="pipeline-node-title">{title}</div>
            <div className="pipeline-node-detail">{detail}</div>
        </div>
    );
}

export function Skeleton({ className = "", style }) {
    return <span className={`skeleton ${className}`} style={style} aria-hidden="true" />;
}

export function SkeletonBlock({ lines = 3, title = false, chart = false, pills = 0 }) {
    return (
        <div className="skeleton-grid">
            {pills > 0 ? (
                <div className={`stat-grid compact-grid`} style={{ gridTemplateColumns: `repeat(${Math.min(pills, 4)}, minmax(0, 1fr))` }}>
                    {Array.from({ length: pills }).map((_, i) => (
                        <Skeleton key={i} className="skeleton-pill" />
                    ))}
                </div>
            ) : null}
            {title ? <Skeleton className="skeleton-title" /> : null}
            {chart ? <Skeleton className="skeleton-chart" /> : null}
            {!chart && Array.from({ length: lines }).map((_, i) => (
                <Skeleton key={i} className={`skeleton-line ${i === lines - 1 ? "skeleton-short" : ""}`} />
            ))}
        </div>
    );
}

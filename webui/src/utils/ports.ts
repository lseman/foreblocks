// src/utils/ports.ts

// Keep these in sync with NodeCard width + paddings
export const CARD_WIDTH = 260; // Must match NodeCard's inline width
export const CARD_PADDING_X = 12; // Tailwind p-3 => 12px left/right
export const HEADER_H = 56; // Fixed header height for geometry stability
export const PORT_ROW_H = 26; // Height of each port row
export const PORT_ROW_GAP = 8; // Tailwind space-y-2 => 8px vertical gap between rows
export const PORT_OUTSET = 20; // Tailwind -left-5 / -right-5 => 20px
export const PORT_DIAMETER = 14; // Tailwind w-3.5 => 14px
export const PORT_RADIUS = PORT_DIAMETER / 2;

// Y of the first port row CENTER relative to card top:
// header height + ports container padding-top (px) + half row height
export const PORT_BASE_Y = HEADER_H + 12 + PORT_ROW_H / 2;

export type PortSide = "input" | "output";

export interface Position {
    x: number;
    y: number;
}
export interface PositionedNode {
    position: Position;
}

/**
 * Compute the exact center coordinates of a port circle.
 * Accounts for fixed row height AND Tailwind `space-y-2` gaps between rows.
 *
 * @param node Node with absolute position
 * @param side "input" | "output"
 * @param rowIndex 0-based index within its block (inputs or outputs)
 * @param numInputsBeforeThisBlock How many input rows appear before this block (for outputs)
 */
export function getPortCenter(
    node: PositionedNode,
    side: PortSide,
    rowIndex: number,
    numInputsBeforeThisBlock = 0,
): Position {
    const rowsBefore = numInputsBeforeThisBlock + rowIndex;
    const step = PORT_ROW_H + PORT_ROW_GAP; // one row plus its gap below it
    const y = node.position.y + PORT_BASE_Y + rowsBefore * step;

    const x =
        side === "input"
            ? node.position.x + CARD_PADDING_X - PORT_OUTSET + PORT_RADIUS
            : node.position.x +
              CARD_WIDTH -
              CARD_PADDING_X +
              PORT_OUTSET -
              PORT_RADIUS;

    return { x, y };
}

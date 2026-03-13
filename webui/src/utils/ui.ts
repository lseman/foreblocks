
export const tailwindClassToGradientStyle = (colorClass?: string): string => {
  if (!colorClass || colorClass === "bg-slate-700/60") {
    return "linear-gradient(to bottom right, rgb(51 65 85 / 0.6), rgb(15 23 42 / 0.6))";
  }
  const directionMap: Record<string, string> = {
    "to-br": "to bottom right",
    "to-tr": "to top right",
    "to-bl": "to bottom left",
    "to-tl": "to top left",
    "to-r": "to right",
    "to-l": "to left",
    "to-t": "to top",
    "to-b": "to bottom",
  };
  const colorMap: Record<string, Record<number, string>> = {
    slate: { 700: "rgb(51 65 85)", 800: "rgb(15 23 42)" },
    green: { 700: "rgb(4 120 87)", 800: "rgb(6 95 70)" },
    blue:  { 700: "rgb(29 78 216)", 800: "rgb(30 64 175)" },
    purple:{ 700: "rgb(147 51 234)", 800: "rgb(124 58 237)" },
  };
  const directionMatch = colorClass.match(/bg-gradient-to-([a-z-]+)/);
  const dir = directionMatch ? directionMap[directionMatch[1]] || "to bottom right" : "to bottom right";
  const fromMatch = colorClass.match(/from-([a-z]+)-(\d+)/);
  const toMatch   = colorClass.match(/to-([a-z]+)-(\d+)/);
  const fromColor = fromMatch ? (colorMap[fromMatch[1]]?.[parseInt(fromMatch[2])]) || "rgb(51 65 85)" : "rgb(51 65 85)";
  const toColor   = toMatch   ? (colorMap[toMatch[1]]?.[parseInt(toMatch[2])])   || "rgb(15 23 42)"  : "rgb(15 23 42)";
  return `linear-gradient(${dir}, ${fromColor}, ${toColor})`;
};

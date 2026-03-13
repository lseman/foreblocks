import { createRoot } from "react-dom/client";
import TimeSeriesNodeEditor from "./TimeSeriesNodeEditor";
import './index.css';

const container = document.getElementById("root");
if (container) {
    const root = createRoot(container);
    root.render(<TimeSeriesNodeEditor />);
}
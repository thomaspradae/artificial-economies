import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App";
import { loadCrossWorldBundle, loadPricingCapScenario } from "./engine/ScenarioLoader";
import "./styles.css";

const root = createRoot(document.getElementById("root") as HTMLElement);

try {
  const [initialBundle, crossWorldBundle] = await Promise.all([
    loadPricingCapScenario(),
    loadCrossWorldBundle()
  ]);
  root.render(
    <StrictMode>
      <App initialBundle={initialBundle} crossWorldBundle={crossWorldBundle} />
    </StrictMode>
  );
} catch (error) {
  root.render(
    <StrictMode>
      <main className="loading">Failed to load scenario: {String(error)}</main>
    </StrictMode>
  );
}

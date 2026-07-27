import { useEffect, useRef } from "react";
import { Application, Container, Graphics, Text } from "pixi.js";
import type { SimulationFrame } from "../../types/scenario";

interface PricingSceneProps {
  noneFrame: SimulationFrame;
  capFrame: SimulationFrame;
  priceCap: number;
}

interface SceneState {
  app: Application;
  stageLayer: Container;
}

function colorForPrice(price: number): number {
  if (price <= 4) return 0x2f9e44;
  if (price <= 5) return 0xd19a00;
  if (price <= 7) return 0xd46b08;
  return 0xc92a2a;
}

export function PricingScene({ noneFrame, capFrame, priceCap }: PricingSceneProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const stateRef = useRef<SceneState | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function setup() {
      if (!hostRef.current) return;
      const app = new Application();
      await app.init({
        width: hostRef.current.clientWidth,
        height: 430,
        antialias: true,
        backgroundAlpha: 0,
        resolution: Math.min(window.devicePixelRatio || 1, 2),
        autoDensity: true
      });
      if (cancelled || !hostRef.current) {
        app.destroy();
        return;
      }
      const stageLayer = new Container();
      app.stage.addChild(stageLayer);
      hostRef.current.appendChild(app.canvas);
      stateRef.current = { app, stageLayer };
      renderScene(stageLayer, app.canvas.width / app.renderer.resolution, 430, noneFrame, capFrame, priceCap);
    }
    setup();
    return () => {
      cancelled = true;
      if (stateRef.current) {
        stateRef.current.app.destroy(true);
        stateRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    renderScene(
      state.stageLayer,
      state.app.canvas.width / state.app.renderer.resolution,
      430,
      noneFrame,
      capFrame,
      priceCap
    );
  }, [noneFrame, capFrame, priceCap]);

  return <div className="pricing-scene" ref={hostRef} aria-label="Animated Pricing Arena replay" />;
}

function renderScene(
  layer: Container,
  width: number,
  height: number,
  noneFrame: SimulationFrame,
  capFrame: SimulationFrame,
  priceCap: number
) {
  layer.removeChildren();
  const background = new Graphics();
  background.roundRect(0, 0, width, height, 8).fill(0xf6f4ed).stroke({ color: 0x2a2925, width: 1 });
  layer.addChild(background);

  const gap = 18;
  const panelWidth = (width - gap - 24) / 2;
  drawMarket(layer, 12, 12, panelWidth, height - 24, "No regulation", noneFrame, false, priceCap);
  drawMarket(layer, 12 + panelWidth + gap, 12, panelWidth, height - 24, "Price cap", capFrame, true, priceCap);
}

function drawMarket(
  layer: Container,
  x: number,
  y: number,
  width: number,
  height: number,
  title: string,
  frame: SimulationFrame,
  capEnabled: boolean,
  priceCap: number
) {
  const panel = new Graphics();
  panel.roundRect(x, y, width, height, 8).fill(0xfffdf7).stroke({ color: capEnabled ? 0xa83232 : 0x2f4f4f, width: 2 });
  layer.addChild(panel);

  const titleText = new Text({
    text: title,
    style: { fill: 0x1c1b18, fontSize: 16, fontWeight: "700" }
  });
  titleText.position.set(x + 14, y + 12);
  layer.addChild(titleText);

  if (capEnabled) {
    const capBand = new Graphics();
    capBand.roundRect(x + width - 98, y + 10, 82, 24, 6).fill(0xffe3e3).stroke({ color: 0xa83232, width: 1 });
    layer.addChild(capBand);
    const capText = new Text({
      text: `cap ${priceCap.toFixed(1)}`,
      style: { fill: 0x7c1f1f, fontSize: 12, fontWeight: "700" }
    });
    capText.position.set(x + width - 84, y + 16);
    layer.addChild(capText);
  }

  const firmY = y + 92;
  const firmA = { x: x + width * 0.28, y: firmY };
  const firmB = { x: x + width * 0.72, y: firmY };
  drawFirm(layer, firmA.x, firmA.y, "Firm A", frame.agents[0]);
  drawFirm(layer, firmB.x, firmB.y, "Firm B", frame.agents[1]);

  const poolY = y + height - 120;
  const totalQuantity = Math.max(1, frame.metrics.quantity);
  const q1Share = frame.agents[0].quantity / totalQuantity;
  const consumers = 70;
  const q1Count = Math.round(consumers * q1Share);
  for (let index = 0; index < consumers; index += 1) {
    const towardA = index < q1Count;
    const t = (index % 14) / 13;
    const row = Math.floor(index / 14);
    const baseX = x + 34 + t * (width - 68);
    const baseY = poolY + row * 13;
    const target = towardA ? firmA : firmB;
    const pull = 0.22 + ((index * 17) % 19) / 100;
    const cx = baseX * (1 - pull) + target.x * pull;
    const cy = baseY * (1 - pull) + (target.y + 36) * pull;
    const dot = new Graphics();
    dot.circle(cx, cy, 3.2).fill(towardA ? 0x2f6f9f : 0x8b5cf6);
    layer.addChild(dot);
  }

  const flow = new Graphics();
  flow.moveTo(x + width / 2, poolY - 22)
    .lineTo(firmA.x, firmA.y + 42)
    .stroke({ color: 0x2f6f9f, width: Math.max(1.5, frame.agents[0].quantity / 18), alpha: 0.42 });
  flow.moveTo(x + width / 2, poolY - 22)
    .lineTo(firmB.x, firmB.y + 42)
    .stroke({ color: 0x8b5cf6, width: Math.max(1.5, frame.agents[1].quantity / 18), alpha: 0.42 });
  layer.addChild(flow);

  if (frame.events.length > 0) {
    const alert = new Graphics();
    alert.roundRect(x + 16, y + 44, width - 32, 30, 6).fill(0xfff0d7).stroke({ color: 0xd46b08, width: 1 });
    layer.addChild(alert);
    const event = frame.events[0];
    const label = new Text({
      text: `${event.agentId.replace("_", " ")} clipped: ${event.requested.toFixed(1)} -> ${event.executed.toFixed(1)}`,
      style: { fill: 0x7a3d00, fontSize: 12, fontWeight: "700" }
    });
    label.position.set(x + 26, y + 52);
    layer.addChild(label);
  }

  const footer = new Text({
    text: `price ${frame.metrics.price.toFixed(2)}   quantity ${frame.metrics.quantity.toFixed(1)}   profit ${frame.metrics.profit.toFixed(1)}`,
    style: { fill: 0x2c2a25, fontSize: 13, fontWeight: "600" }
  });
  footer.position.set(x + 14, y + height - 30);
  layer.addChild(footer);
}

function drawFirm(
  layer: Container,
  x: number,
  y: number,
  label: string,
  agent: SimulationFrame["agents"][number]
) {
  const building = new Graphics();
  building.roundRect(x - 45, y - 24, 90, 64, 7)
    .fill(0xffffff)
    .stroke({ color: colorForPrice(agent.action.price), width: 4 });
  building.rect(x - 34, y - 10, 16, 14).fill(0xdde7e8);
  building.rect(x - 8, y - 10, 16, 14).fill(0xdde7e8);
  building.rect(x + 18, y - 10, 16, 14).fill(0xdde7e8);
  building.rect(x - 8, y + 16, 16, 24).fill(0x6c584c);
  layer.addChild(building);

  const name = new Text({
    text: label,
    style: { fill: 0x1c1b18, fontSize: 12, fontWeight: "700" }
  });
  name.anchor.set(0.5, 0);
  name.position.set(x, y - 48);
  layer.addChild(name);

  const price = new Text({
    text: `$${agent.action.price.toFixed(1)}`,
    style: { fill: colorForPrice(agent.action.price), fontSize: 18, fontWeight: "800" }
  });
  price.anchor.set(0.5, 0);
  price.position.set(x, y + 46);
  layer.addChild(price);

  const profit = new Text({
    text: `profit ${agent.profit.toFixed(0)}`,
    style: { fill: 0x4f4a43, fontSize: 11 }
  });
  profit.anchor.set(0.5, 0);
  profit.position.set(x, y + 68);
  layer.addChild(profit);
}

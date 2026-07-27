export interface TimelineState {
  frameIndex: number;
  playing: boolean;
  speed: number;
}

export function nextFrameIndex(current: number, frameCount: number): number {
  if (frameCount <= 0) return 0;
  return (current + 1) % frameCount;
}

export function clampFrameIndex(value: number, frameCount: number): number {
  if (frameCount <= 0) return 0;
  return Math.min(Math.max(0, value), frameCount - 1);
}

import type { Texture } from "three";
import { CanvasTexture } from "three";

export default function (): Texture {
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    const context = canvas.getContext("2d")!;

    const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
    gradient.addColorStop(0.2, "rgba(255, 255, 255, 0.8)");
    gradient.addColorStop(0.4, "rgba(255, 255, 255, 0.4)");
    gradient.addColorStop(0.7, "rgba(255, 255, 255, 0.1)");
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

    context.fillStyle = gradient;
    context.fillRect(0, 0, 32, 32);

    const texture = new CanvasTexture(canvas);
    return texture;
}

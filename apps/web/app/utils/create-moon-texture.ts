import type { Texture } from "three";
import { CanvasTexture, RepeatWrapping } from "three";

interface MoonColors {
    base: string[];
    land: string[];
    poles?: string;
}

export default function (colorScheme: MoonColors): Texture {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d")!;

    const gradient = ctx.createLinearGradient(0, 0, 256, 256);
    colorScheme.base.forEach((color, index) => {
        gradient.addColorStop(index / (colorScheme.base.length - 1), color);
    });

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 256);

    const spotCount = Math.floor(Math.random() * 5) + 3;
    for (let i = 0; i < spotCount; i++) {
        const x = Math.random() * 256;
        const y = Math.random() * 256;
        const radius = Math.random() * 50 + 20;
        const landColor = colorScheme.land[Math.floor(Math.random() * colorScheme.land.length)];

        if (landColor) {
            ctx.fillStyle = landColor;
        }

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    if (colorScheme.poles) {
        ctx.fillStyle = colorScheme.poles;
        ctx.beginPath();
        ctx.arc(128, 20, 25, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(128, 236, 22, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    for (let i = 0; i < 10; i++) {
        const x = Math.random() * 256;
        const y = Math.random() * 256;
        const radius = Math.random() * 8 + 3;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    const texture = new CanvasTexture(canvas);
    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;

    return texture;
}

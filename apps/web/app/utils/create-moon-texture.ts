import type { Texture } from "three";
import { CanvasTexture, RepeatWrapping } from "three";

interface MoonColors {
    base: string[];
    land: string[];
    poles?: string;
}

/**
 * Creates a realistic moon/celestial body texture with craters and surface details
 */
export default function (colorScheme: MoonColors): Texture {
    const canvas = document.createElement("canvas");
    const size = 256;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    // Base gradient with more variation
    const gradient = ctx.createRadialGradient(
        size * 0.4, size * 0.4, 0,
        size * 0.5, size * 0.5, size * 0.7
    );
    colorScheme.base.forEach((color, index) => {
        gradient.addColorStop(index / (colorScheme.base.length - 1), color);
    });
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    // Add surface noise/texture for realism
    for (let i = 0; i < 80; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radius = Math.random() * 15 + 3;
        const alpha = Math.random() * 0.15 + 0.05;
        const isDark = Math.random() > 0.5;

        ctx.fillStyle = isDark
            ? `rgba(0, 0, 0, ${alpha})`
            : `rgba(255, 255, 255, ${alpha * 0.5})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Maria (dark regions) - larger land features
    const mariaCount = Math.floor(Math.random() * 4) + 2;
    for (let i = 0; i < mariaCount; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radiusX = Math.random() * 60 + 30;
        const radiusY = Math.random() * 40 + 20;
        const rotation = Math.random() * Math.PI;
        const landColor = colorScheme.land[Math.floor(Math.random() * colorScheme.land.length)]!;

        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation);

        const mariaGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radiusX);
        mariaGradient.addColorStop(0, landColor);
        mariaGradient.addColorStop(0.7, landColor);
        mariaGradient.addColorStop(1, "transparent");

        ctx.fillStyle = mariaGradient;
        ctx.beginPath();
        ctx.ellipse(0, 0, radiusX, radiusY, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    // Impact craters with rim lighting
    const craterCount = Math.floor(Math.random() * 12) + 8;
    for (let i = 0; i < craterCount; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radius = Math.random() * 18 + 5;

        // Crater shadow (dark interior)
        const craterGradient = ctx.createRadialGradient(
            x - radius * 0.2, y - radius * 0.2, 0,
            x, y, radius
        );
        craterGradient.addColorStop(0, "rgba(0, 0, 0, 0.5)");
        craterGradient.addColorStop(0.6, "rgba(0, 0, 0, 0.3)");
        craterGradient.addColorStop(1, "rgba(0, 0, 0, 0)");

        ctx.fillStyle = craterGradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Crater rim highlight
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 + Math.random() * 0.15})`;
        ctx.lineWidth = 1 + Math.random();
        ctx.beginPath();
        ctx.arc(x, y, radius * 0.95, Math.PI * 0.8, Math.PI * 2.2);
        ctx.stroke();
    }

    // Small craterlets
    for (let i = 0; i < 25; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radius = Math.random() * 4 + 1;

        ctx.fillStyle = `rgba(0, 0, 0, ${0.2 + Math.random() * 0.2})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Polar regions if specified
    if (colorScheme.poles) {
        ctx.fillStyle = colorScheme.poles;

        // North pole with soft edge
        const northGradient = ctx.createRadialGradient(size / 2, 0, 0, size / 2, 20, 40);
        northGradient.addColorStop(0, colorScheme.poles);
        northGradient.addColorStop(1, "transparent");
        ctx.fillStyle = northGradient;
        ctx.beginPath();
        ctx.ellipse(size / 2, 12, 50, 20, 0, 0, Math.PI * 2);
        ctx.fill();

        // South pole
        const southGradient = ctx.createRadialGradient(size / 2, size, 0, size / 2, size - 20, 35);
        southGradient.addColorStop(0, colorScheme.poles);
        southGradient.addColorStop(1, "transparent");
        ctx.fillStyle = southGradient;
        ctx.beginPath();
        ctx.ellipse(size / 2, size - 10, 45, 18, 0, 0, Math.PI * 2);
        ctx.fill();
    }

    // Mineral deposits / bright spots
    for (let i = 0; i < 5; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radius = Math.random() * 6 + 2;

        ctx.fillStyle = `rgba(255, 255, 255, ${0.3 + Math.random() * 0.2})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    const texture = new CanvasTexture(canvas);
    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;

    return texture;
}

import type { Texture } from "three";
import { CanvasTexture, RepeatWrapping } from "three";

/**
 * Creates a more realistic planet texture using procedural generation
 * Optimized for performance using canvas operations
 */
export default function createPlanetTexture(): Texture {
    const canvas = document.createElement("canvas");
    const size = 512;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    // Base ocean gradient - deep blues
    const oceanGradient = ctx.createRadialGradient(
        size * 0.3, size * 0.3, 0,
        size * 0.5, size * 0.5, size * 0.8
    );
    oceanGradient.addColorStop(0, "#0c4a6e");
    oceanGradient.addColorStop(0.3, "#0369a1");
    oceanGradient.addColorStop(0.6, "#0284c7");
    oceanGradient.addColorStop(1, "#0c4a6e");
    ctx.fillStyle = oceanGradient;
    ctx.fillRect(0, 0, size, size);

    // Add ocean depth variation with noise-like patterns
    for (let i = 0; i < 50; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radius = Math.random() * 80 + 20;
        const alpha = Math.random() * 0.15 + 0.05;

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, `rgba(3, 105, 161, ${alpha})`);
        gradient.addColorStop(1, "rgba(12, 74, 110, 0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Continents - varied greens and browns with naturalish shapes
    const continentData = [
        { x: 120, y: 100, scale: 1.2, rotation: 0 },
        { x: 350, y: 180, scale: 0.9, rotation: Math.PI / 4 },
        { x: 220, y: 320, scale: 1.4, rotation: -Math.PI / 6 },
        { x: 80, y: 280, scale: 0.6, rotation: Math.PI / 3 },
        { x: 420, y: 380, scale: 0.7, rotation: -Math.PI / 4 },
    ];

    continentData.forEach(continent => {
        ctx.save();
        ctx.translate(continent.x, continent.y);
        ctx.rotate(continent.rotation);
        ctx.scale(continent.scale, continent.scale);

        // Main landmass
        const landGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 60);
        landGradient.addColorStop(0, "#15803d");
        landGradient.addColorStop(0.4, "#166534");
        landGradient.addColorStop(0.7, "#14532d");
        landGradient.addColorStop(1, "#052e16");
        ctx.fillStyle = landGradient;

        // Draw organic blob shapes
        ctx.beginPath();
        ctx.moveTo(0, -50);
        ctx.bezierCurveTo(40, -40, 60, -10, 50, 30);
        ctx.bezierCurveTo(40, 60, 10, 70, -20, 50);
        ctx.bezierCurveTo(-50, 30, -60, 0, -40, -30);
        ctx.bezierCurveTo(-30, -50, -10, -55, 0, -50);
        ctx.fill();

        // Add highlands/mountains (brownish areas)
        ctx.fillStyle = "#78350f";
        ctx.beginPath();
        ctx.arc(10, 0, 15, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#92400e";
        ctx.beginPath();
        ctx.arc(-15, 15, 10, 0, Math.PI * 2);
        ctx.fill();

        // Add forests (darker greens)
        ctx.fillStyle = "#064e3b";
        ctx.beginPath();
        ctx.arc(20, -20, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    });

    // Ice caps - polar regions
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";

    // North pole
    ctx.beginPath();
    ctx.ellipse(size / 2, 25, 80, 30, 0, 0, Math.PI * 2);
    ctx.fill();

    // South pole
    ctx.beginPath();
    ctx.ellipse(size / 2, size - 20, 70, 25, 0, 0, Math.PI * 2);
    ctx.fill();

    // Add ice cap texture
    ctx.fillStyle = "rgba(200, 230, 255, 0.4)";
    for (let i = 0; i < 15; i++) {
        const x = size / 2 + (Math.random() - 0.5) * 140;
        const yTop = Math.random() * 40 + 5;
        const yBottom = size - Math.random() * 35 - 5;
        const radius = Math.random() * 10 + 5;

        ctx.beginPath();
        ctx.arc(x, yTop, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, yBottom, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Cloud layer - soft white patches
    ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
    for (let i = 0; i < 25; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const width = Math.random() * 100 + 40;
        const height = Math.random() * 30 + 15;
        const rotation = Math.random() * Math.PI;

        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation);
        ctx.beginPath();
        ctx.ellipse(0, 0, width / 2, height / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    // Spiral storm systems
    const storms = [
        { x: 380, y: 250, size: 25 },
        { x: 150, y: 200, size: 18 },
    ];

    storms.forEach(storm => {
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.beginPath();
        ctx.arc(storm.x, storm.y, storm.size, 0, Math.PI * 2);
        ctx.fill();

        // Spiral arms
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 4;
        ctx.beginPath();
        for (let angle = 0; angle < Math.PI * 4; angle += 0.1) {
            const r = 5 + angle * 3;
            const px = storm.x + Math.cos(angle) * r;
            const py = storm.y + Math.sin(angle) * r;
            if (angle === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    });

    // Atmospheric haze at edges
    const hazeGradient = ctx.createLinearGradient(0, 0, 0, size);
    hazeGradient.addColorStop(0, "rgba(135, 206, 235, 0.2)");
    hazeGradient.addColorStop(0.15, "rgba(135, 206, 235, 0)");
    hazeGradient.addColorStop(0.85, "rgba(135, 206, 235, 0)");
    hazeGradient.addColorStop(1, "rgba(135, 206, 235, 0.2)");
    ctx.fillStyle = hazeGradient;
    ctx.fillRect(0, 0, size, size);

    const texture = new CanvasTexture(canvas);
    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;

    return texture;
}

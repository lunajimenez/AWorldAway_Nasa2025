<script setup lang="ts">
const isLoading = defineModel<boolean>({ default: true });

const startTime = ref(Date.now());

async function hideLoading() {
    const elapsed = Date.now() - startTime.value;
    const remaining = Math.max(0, CONSTANTS.HOME.MIN_LOADING_TIME - elapsed);
    await new Promise((resolve) => setTimeout(resolve, remaining));
    isLoading.value = false;
}

defineExpose({ hideLoading });

import { onMounted, onUnmounted, ref as vueRef } from 'vue';

const canvasRef = vueRef<HTMLCanvasElement | null>(null);

let rafId: number | null = null;
let stars: Array<any> = [];
let asteroids: Array<any> = [];
let ctx: CanvasRenderingContext2D | null = null;
let dpr = 1;
let width = 0;
let height = 0;

function setupCanvas() {
    const canvas = canvasRef.value!;
    if (!canvas) return;
    dpr = Math.max(1, window.devicePixelRatio || 1);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx = canvas.getContext('2d');
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resize() {
    setupCanvas();
}

function createScene() {
    stars = [];
    asteroids = [];
    const starCount = Math.floor((width * height) / 1500) + 150; // scale with area
    for (let i = 0; i < starCount; i++) {
        stars.push({
            x: Math.random() * width,
            y: Math.random() * height,
            r: Math.random() * 1.6 + 0.2,
            alpha: Math.random() * 0.8 + 0.2,
            tw: Math.random() * 0.02 + 0.005,
        });
    }

    const astCount = Math.floor(Math.max(12, (width * height) / 40000));
    for (let i = 0; i < astCount; i++) {
        const size = Math.random() * 6 + 2;
        asteroids.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.6,
            vy: (Math.random() - 0.2) * 0.6 - 0.1,
            size,
            rot: Math.random() * Math.PI * 2,
            rotSpeed: (Math.random() - 0.5) * 0.02,
        });
    }
}

function drawNebula() {
    if (!ctx) return;
    const g = ctx.createLinearGradient(0, 0, width, height);
    g.addColorStop(0, 'rgba(14, 18, 28, 0.7)');
    g.addColorStop(0.5, 'rgba(20, 24, 36, 0.5)');
    g.addColorStop(1, 'rgba(6, 8, 12, 0.85)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    // subtle soft glow
    const cx = width * 0.25 + Math.cos(Date.now() / 60000) * 80;
    const cy = height * 0.3;
    const rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(width, height) * 0.6);
    rg.addColorStop(0, 'rgba(60,90,180,0.06)');
    rg.addColorStop(0.3, 'rgba(40,60,120,0.03)');
    rg.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'source-over';
}

function drawStars(time: number) {
    if (!ctx) return;
    for (const s of stars) {
        s.alpha += s.tw * Math.sin(time / 1000 + s.x + s.y) * 0.02;
        const a = Math.max(0.15, Math.min(1, s.alpha));
        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${a})`;
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawAsteroids(delta: number) {
    if (!ctx) return;
    ctx.fillStyle = '#bfb6a6';
    for (const a of asteroids) {
        a.x += a.vx * delta * 0.06;
        a.y += a.vy * delta * 0.06;
        a.rot += a.rotSpeed * delta * 0.06;
        if (a.x < -50) a.x = width + 50;
        if (a.x > width + 50) a.x = -50;
        if (a.y < -50) a.y = height + 50;
        if (a.y > height + 50) a.y = -50;

        ctx.save();
        ctx.translate(a.x, a.y);
        ctx.rotate(a.rot);
        ctx.beginPath();
        ctx.ellipse(0, 0, a.size * 1.1, a.size * 0.7, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

function drawPlanet(time: number) {
    if (!ctx) return;
    const cx = width * 0.65;
    const cy = height * 0.6;
    const orbitR = Math.min(width, height) * 0.18;
    const angle = (time / 5000) % (Math.PI * 2);
    const px = cx + Math.cos(angle) * orbitR * 0.6;
    const py = cy + Math.sin(angle) * orbitR * 0.35;
    const pr = Math.min(width, height) * 0.08;

    // shadowed planet with gradient
    const g = ctx.createRadialGradient(px - pr * 0.3, py - pr * 0.3, pr * 0.1, px, py, pr);
    g.addColorStop(0, '#9fb4ff');
    g.addColorStop(0.6, '#4666a3');
    g.addColorStop(1, '#102033');
    ctx.beginPath();
    ctx.fillStyle = g;
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fill();

    // subtle atmosphere glow
    ctx.beginPath();
    ctx.fillStyle = 'rgba(120,160,255,0.06)';
    ctx.arc(px, py, pr * 1.25, 0, Math.PI * 2);
    ctx.fill();

    // specular highlight
    ctx.beginPath();
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.ellipse(px - pr * 0.3, py - pr * 0.45, pr * 0.25, pr * 0.12, 0, 0, Math.PI * 2);
    ctx.fill();

    // orbit ring (very faint)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(200,200,220,0.03)';
    ctx.lineWidth = 1;
    ctx.ellipse(cx, cy, orbitR, orbitR * 0.6, 0, 0, Math.PI * 2);
    ctx.stroke();
}

let last = performance.now();
function render(t: number) {
    if (!ctx) return;
    const now = t;
    const delta = now - last;
    last = now;

    ctx.clearRect(0, 0, width, height);
    drawNebula();
    drawStars(now);
    drawAsteroids(delta);
    drawPlanet(now);

    // small vignette
    const vg = ctx.createRadialGradient(width / 2, height / 2, Math.max(width, height) * 0.2, width / 2, height / 2, Math.max(width, height) * 0.8);
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(1, 'rgba(0,0,0,0.35)');
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, width, height);

    // schedule
    if (document.visibilityState === 'visible') {
        rafId = requestAnimationFrame(render);
    } else {
        rafId = null;
    }
}

function start() {
    setupCanvas();
    createScene();
    last = performance.now();
    if (!rafId) rafId = requestAnimationFrame(render);
}

function stop() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
}

onMounted(() => {
    start();
    window.addEventListener('resize', resize);
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible' && !rafId) {
            last = performance.now();
            rafId = requestAnimationFrame(render);
        }
    });
});

onUnmounted(() => {
    stop();
    window.removeEventListener('resize', resize);
});
</script>

<template>
    <Transition name="fade">
        <div
            v-if="isLoading"
            class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900"
        >
            <!-- canvas detrás de todo -->
            <canvas ref="canvasRef" class="absolute inset-0 w-full h-full pointer-events-none" />

            <div class="relative z-10 flex flex-col items-center justify-center">
                <div class="relative mb-8">
                    <div class="absolute inset-0 animate-pulse blur-2xl bg-primary/30 rounded-full" />
                    <div class="relative size-32 rounded-full bg-primary/10 flex items-center justify-center">
                        <svg
                            class="size-20 text-primary animate-spin"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                            <path
                                class="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            />
                        </svg>
                    </div>
                </div>

                <h1 class="text-3xl font-bold mb-2 tracking-wider font-pixelify text-white">A WORLD AWAY</h1>
            <p class="text-sm animate-pulse text-white/70">{{ $t('pages.home.loading.slogan') }}</p>
            </div>
        </div>
    </Transition>
</template>

<style scoped>
    .fade-enter-active,
    .fade-leave-active {
        transition: opacity 0.5s ease;
    }

    .fade-enter-from,
    .fade-leave-to {
        opacity: 0;
    }

    /* sizes used in markup (keeps existing CSS tokens used elsewhere) */
    .size-20 {
        width: 5rem;
        height: 5rem;
    }
    .size-32 {
        width: 8rem;
        height: 8rem;
    }
</style>

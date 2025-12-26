import {
    AdditiveBlending,
    AmbientLight,
    BufferAttribute,
    BufferGeometry,
    CanvasTexture,
    DirectionalLight,
    Line,
    LineBasicMaterial,
    Mesh,
    MeshPhongMaterial,
    PCFSoftShadowMap,
    PerspectiveCamera,
    PointLight,
    Points,
    PointsMaterial,
    Scene,
    SphereGeometry,
    Spherical,
    SpriteMaterial,
    Sprite,
    Vector3,
    WebGLRenderer,
} from "three";
import createPlanetTexture from "~/utils/create-planet-texture";

// Floating animation state
let floatTime = 0;
const FLOAT_AMPLITUDE = 0.08;
const FLOAT_SPEED = 0.8;

// Nebula configuration - vibrant space colors like gas clouds illuminated by stars
const NEBULA_COLORS = [
    // Warm nebula colors (like emission nebulae)
    { r: 255, g: 100, b: 150, a: 0.04 },   // Hot pink (hydrogen alpha)
    { r: 255, g: 140, b: 50, a: 0.035 },   // Electric orange
    { r: 255, g: 80, b: 80, a: 0.03 },     // Crimson red
    { r: 255, g: 200, b: 100, a: 0.025 },  // Golden yellow

    // Cool nebula colors (like reflection nebulae)
    { r: 100, g: 200, b: 255, a: 0.04 },   // Cosmic cyan
    { r: 80, g: 100, b: 255, a: 0.035 },   // Deep blue
    { r: 150, g: 50, b: 255, a: 0.04 },    // Vivid purple
    { r: 255, g: 50, b: 200, a: 0.035 },   // Magenta

    // Exotic colors (like planetary nebulae)
    { r: 50, g: 255, b: 150, a: 0.03 },    // Emerald green (OIII emission)
    { r: 200, g: 255, b: 100, a: 0.025 },  // Lime green
    { r: 0, g: 255, b: 200, a: 0.03 },     // Teal
    { r: 255, g: 150, b: 255, a: 0.03 },   // Soft pink
];

/**
 * Creates a nebula texture using canvas gradients with more detail
 */
function createNebulaTexture(color: { r: number; g: number; b: number; a: number }, size: number): CanvasTexture {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    // Create main radial gradient for soft nebula effect
    const gradient = ctx.createRadialGradient(
        size / 2, size / 2, 0,
        size / 2, size / 2, size / 2
    );

    gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 1.5})`);
    gradient.addColorStop(0.2, `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`);
    gradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.5})`);
    gradient.addColorStop(0.8, `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.2})`);
    gradient.addColorStop(1, "rgba(0, 0, 0, 0)");

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    // Add wispy tendrils for more realistic nebula look
    for (let i = 0; i < 5; i++) {
        const tendrilGradient = ctx.createRadialGradient(
            size / 2 + (Math.random() - 0.5) * size * 0.4,
            size / 2 + (Math.random() - 0.5) * size * 0.4,
            0,
            size / 2,
            size / 2,
            size * 0.4
        );
        tendrilGradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.8})`);
        tendrilGradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.3})`);
        tendrilGradient.addColorStop(1, "rgba(0, 0, 0, 0)");

        ctx.fillStyle = tendrilGradient;
        ctx.beginPath();
        ctx.ellipse(
            size / 2 + (Math.random() - 0.5) * size * 0.3,
            size / 2 + (Math.random() - 0.5) * size * 0.3,
            size * (0.2 + Math.random() * 0.2),
            size * (0.1 + Math.random() * 0.15),
            Math.random() * Math.PI,
            0,
            Math.PI * 2
        );
        ctx.fill();
    }

    // Add cosmic dust particles
    for (let i = 0; i < 20; i++) {
        const dustX = Math.random() * size;
        const dustY = Math.random() * size;
        const dustRadius = Math.random() * 8 + 2;
        const dustAlpha = color.a * (0.3 + Math.random() * 0.5);

        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${dustAlpha})`;
        ctx.beginPath();
        ctx.arc(dustX, dustY, dustRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    const texture = new CanvasTexture(canvas);
    return texture;
}

/**
 * Creates layered nebula sprites for depth with vibrant colors
 */
function createNebulae(scene: Scene): void {
    const nebulaCount = 18; // More nebulae for colorful space

    for (let i = 0; i < nebulaCount; i++) {
        const color = NEBULA_COLORS[i % NEBULA_COLORS.length]!;
        const size = 256 + Math.random() * 384;
        const texture = createNebulaTexture(color, size);

        const material = new SpriteMaterial({
            map: texture,
            transparent: true,
            blending: AdditiveBlending,
            depthWrite: false,
        });

        const sprite = new Sprite(material);

        // Position nebulae at different depths in the background
        const distance = 60 + Math.random() * 140;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        sprite.position.x = distance * Math.sin(phi) * Math.cos(theta);
        sprite.position.y = distance * Math.sin(phi) * Math.sin(theta);
        sprite.position.z = distance * Math.cos(phi);

        // Larger scale for more prominent nebulae
        const scale = 50 + Math.random() * 80;
        sprite.scale.set(scale, scale, 1);

        scene.add(sprite);
    }

    // Add some very large, subtle background nebulae for depth
    const backgroundColors = [
        { r: 80, g: 40, b: 120, a: 0.015 },    // Deep purple haze
        { r: 120, g: 60, b: 80, a: 0.012 },    // Dark rose
        { r: 40, g: 80, b: 100, a: 0.015 },    // Deep teal
    ];

    for (let i = 0; i < 5; i++) {
        const color = backgroundColors[i % backgroundColors.length]!;
        const size = 512;
        const texture = createNebulaTexture(color, size);

        const material = new SpriteMaterial({
            map: texture,
            transparent: true,
            blending: AdditiveBlending,
            depthWrite: false,
        });

        const sprite = new Sprite(material);

        const distance = 150 + Math.random() * 50;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;

        sprite.position.x = distance * Math.sin(phi) * Math.cos(theta);
        sprite.position.y = distance * Math.sin(phi) * Math.sin(theta);
        sprite.position.z = distance * Math.cos(phi);

        // Very large scale for background atmosphere
        const scale = 100 + Math.random() * 80;
        sprite.scale.set(scale, scale, 1);

        scene.add(sprite);
    }
}

export default function () {
    const renderer = shallowRef<WebGLRenderer>();
    const scene = shallowRef<Scene>();
    const cameraReference = shallowRef<PerspectiveCamera>();
    const planetElement = shallowRef<Mesh>();
    const moonsReference = shallowRef<
        Array<{
            mesh: Mesh;
            orbitSpeed: number;
            orbitRadius: number;
            orbitAngle: number;
        }>
    >([]);

    // Comets passing through space
    const cometsReference = shallowRef<
        Array<{
            head: Mesh;
            tail: Line;
            progress: number;
            speed: number;
            startPoint: Vector3;
            endPoint: Vector3;
            active: boolean;
        }>
    >([]);

    // Bright twinkling stars
    const twinklingStarsReference = shallowRef<
        Array<{
            sprite: Sprite;
            baseScale: number;
            twinkleSpeed: number;
            twinkleOffset: number;
        }>
    >([]);

    // Asteroids floating in space
    const asteroidsReference = shallowRef<
        Array<{
            mesh: Mesh;
            rotationSpeed: { x: number; y: number; z: number };
            orbitSpeed: number;
            orbitRadius: number;
            orbitAngle: number;
            orbitTilt: number;
            floatOffset: number;
        }>
    >([]);

    // Shooting stars (fast streaks across the sky)
    const shootingStarsReference = shallowRef<
        Array<{
            line: Line;
            progress: number;
            speed: number;
            startPoint: Vector3;
            endPoint: Vector3;
            active: boolean;
        }>
    >([]);

    // Distant galaxies in the background
    const galaxiesReference = shallowRef<
        Array<{
            sprite: Sprite;
            rotationSpeed: number;
        }>
    >([]);

    // Distant explosions (supernovas)
    const explosionsReference = shallowRef<
        Array<{
            sprite: Sprite;
            progress: number;
            maxScale: number;
            position: Vector3;
            color: { r: number; g: number; b: number };
            active: boolean;
        }>
    >([]);

    const animationIdReference = ref<number>();

    const mouseReference = reactive({
        x: 0,
        y: 0,
        meta: {
            isDown: false,
        },
    });

    const keysReference = reactive({
        w: false,
        a: false,
        s: false,
        d: false,
        shift: false,
        space: false,
    });

    // Auto-orbit camera animation state
    const autoOrbitState = reactive({
        enabled: true,
        orbitAngle: 0,
        verticalAngle: 0,
        idleTime: 0,
        idleThreshold: 3, // Seconds before auto-orbit activates
        isUserInteracting: false,
    });

    const MouseEvent = {
        handleMouseDown(event: MouseEvent) {
            mouseReference.meta.isDown = true;
            mouseReference.x = event.clientX;
            mouseReference.y = event.clientY;
            // User is interacting, disable auto-orbit
            autoOrbitState.isUserInteracting = true;
            autoOrbitState.idleTime = 0;
        },
        handleMouseUp(_event: MouseEvent) {
            mouseReference.meta.isDown = false;
            autoOrbitState.isUserInteracting = false;
        },
        handleMouseMove(event: MouseEvent) {
            if (!mouseReference.meta.isDown || !cameraReference.value) {
                return;
            }

            // Reset idle time on interaction
            autoOrbitState.idleTime = 0;

            const deltaX = event.clientX - mouseReference.x;
            const deltaY = event.clientY - mouseReference.y;

            const spherical = new Spherical();
            spherical.setFromVector3(cameraReference.value.position);

            spherical.theta -= deltaX * 0.01;
            spherical.phi += deltaY * 0.01;
            spherical.phi = Math.max(0.1, 1, Math.min(Math.PI - 0.1, spherical.phi));

            cameraReference.value.position.setFromSpherical(spherical);
            cameraReference.value.lookAt(0, 0, 0);

            // Sync auto-orbit angles with current camera position
            autoOrbitState.orbitAngle = spherical.theta;
            autoOrbitState.verticalAngle = spherical.phi;

            mouseReference.x = event.clientX;
            mouseReference.y = event.clientY;
        },
        handleMouseWheel(event: WheelEvent) {
            if (!cameraReference.value) {
                return;
            }

            // Reset idle time on interaction
            autoOrbitState.idleTime = 0;

            const distance = cameraReference.value.position.length();

            cameraReference.value.position
                .normalize()
                .multiplyScalar(Math.max(2, Math.min(20, distance + event.deltaY * 0.01)));
        },
        handleContextMenu(event: MouseEvent) {
            event.preventDefault();
        },
    };

    const KeyboardEvent = {
        handleKeyDown(event: KeyboardEvent) {
            // Reset idle time on any key press
            autoOrbitState.idleTime = 0;
            autoOrbitState.isUserInteracting = true;

            switch (event.code) {
                case "KeyW": {
                    keysReference.w = true;
                    break;
                }

                case "KeyA": {
                    keysReference.a = true;
                    break;
                }

                case "KeyS": {
                    keysReference.s = true;
                    break;
                }

                case "KeyD": {
                    keysReference.d = true;
                    break;
                }

                case "ShiftLeft":
                case "ShiftRight": {
                    keysReference.shift = true;
                    break;
                }

                case "Space": {
                    keysReference.space = true;
                    event.preventDefault();
                    break;
                }
            }
        },
        handleKeyUp(event: KeyboardEvent) {
            switch (event.code) {
                case "KeyW": {
                    keysReference.w = false;
                    break;
                }

                case "KeyA": {
                    keysReference.a = false;
                    break;
                }

                case "KeyS": {
                    keysReference.s = false;
                    break;
                }

                case "KeyD": {
                    keysReference.d = false;
                    break;
                }

                case "ShiftLeft":
                case "ShiftRight": {
                    keysReference.shift = false;
                    break;
                }

                case "Space": {
                    keysReference.space = false;
                    event.preventDefault();
                    break;
                }
            }

            // Check if all movement keys are released
            if (!keysReference.w && !keysReference.a && !keysReference.s &&
                !keysReference.d && !keysReference.space && !keysReference.shift) {
                autoOrbitState.isUserInteracting = false;
            }
        },
    };

    const Camera = {
        update() {
            if (!cameraReference.value) {
                return;
            }

            const moveSpeed = 0.1;
            const fastSpeed = moveSpeed * 2;

            const forward = new Vector3();
            cameraReference.value.getWorldDirection(forward);

            const right = new Vector3();
            right.crossVectors(forward, cameraReference.value.up).normalize();

            const up = cameraReference.value.up.clone();

            const movement = new Vector3();

            if (keysReference.w) {
                movement.add(forward);
            }

            if (keysReference.s) {
                movement.sub(forward);
            }

            if (keysReference.a) {
                movement.sub(right);
            }

            if (keysReference.d) {
                movement.add(right);
            }

            if (keysReference.space) {
                movement.add(up);
            }

            if (keysReference.shift) {
                movement.sub(up);
            }

            const speed = keysReference.shift && !keysReference.space ? fastSpeed : moveSpeed;
            movement.multiplyScalar(speed);

            cameraReference.value.position.add(movement);
        },
        reset() {
            if (!cameraReference.value) {
                return;
            }

            cameraReference.value.position.set(0, 0, 5);
            cameraReference.value.lookAt(0, 0, 0);

            cameraReference.value.rotation.set(0, 0, 0);
            cameraReference.value.updateMatrixWorld();
        },
    };

    const Window = {
        handleResize() {
            if (!cameraReference.value || !renderer.value) {
                return;
            }

            cameraReference.value.aspect = window.innerWidth / window.innerHeight;
            cameraReference.value.updateProjectionMatrix();
            renderer.value.setSize(window.innerWidth, window.innerHeight);
        },
    };

    function animate() {
        animationIdReference.value = requestAnimationFrame(animate);

        Camera.update();

        // Auto-orbit camera animation
        if (autoOrbitState.enabled && cameraReference.value) {
            if (!autoOrbitState.isUserInteracting) {
                // Increment idle time
                autoOrbitState.idleTime += 0.016;

                // Start auto-orbit after idle threshold
                if (autoOrbitState.idleTime > autoOrbitState.idleThreshold) {
                    // Smooth orbit around the scene with varying speed
                    const orbitSpeedBase = 0.004;
                    autoOrbitState.orbitAngle += orbitSpeedBase;

                    // Subtle vertical wave motion
                    autoOrbitState.verticalAngle += Math.sin(floatTime * 0.15) * 0.0005;

                    // Keep vertical angle in bounds
                    autoOrbitState.verticalAngle = Math.max(0.8, Math.min(2.2, autoOrbitState.verticalAngle));

                    // === DEPTH EFFECT WITH INTERVALS ===
                    // First moves away in steps, then approaches in steps
                    const cycleDuration = 20; // Total cycle duration in seconds
                    const cycleProgress = (floatTime % cycleDuration) / cycleDuration; // 0 to 1

                    const minRadius = 3.5;  // Closest to planet
                    const maxRadius = 18;   // Farthest from planet - shows more of space
                    const radiusRange = maxRadius - minRadius;

                    // Number of steps/intervals
                    const steps = 4;

                    let targetRadius: number;

                    if (cycleProgress < 0.5) {
                        // First half: moving AWAY from planet in steps
                        const awayProgress = cycleProgress * 2; // 0 to 1 during first half
                        const currentStep = Math.floor(awayProgress * steps);
                        const stepProgress = (awayProgress * steps) % 1; // Progress within current step

                        // Smooth transition within each step (ease in-out)
                        const smoothStep = stepProgress < 0.5
                            ? 2 * stepProgress * stepProgress
                            : 1 - Math.pow(-2 * stepProgress + 2, 2) / 2;

                        const stepStart = currentStep / steps;
                        const stepEnd = (currentStep + 1) / steps;
                        const interpolatedStep = stepStart + smoothStep * (stepEnd - stepStart);

                        targetRadius = minRadius + interpolatedStep * radiusRange;
                    } else {
                        // Second half: moving BACK TOWARDS planet in steps
                        const backProgress = (cycleProgress - 0.5) * 2; // 0 to 1 during second half
                        const currentStep = Math.floor(backProgress * steps);
                        const stepProgress = (backProgress * steps) % 1;

                        // Smooth transition within each step
                        const smoothStep = stepProgress < 0.5
                            ? 2 * stepProgress * stepProgress
                            : 1 - Math.pow(-2 * stepProgress + 2, 2) / 2;

                        const stepStart = 1 - (currentStep / steps);
                        const stepEnd = 1 - ((currentStep + 1) / steps);
                        const interpolatedStep = stepStart + smoothStep * (stepEnd - stepStart);

                        targetRadius = minRadius + interpolatedStep * radiusRange;
                    }

                    // Add very subtle breathing effect
                    const breathingEffect = Math.sin(floatTime * 0.5) * 0.15;
                    const finalRadius = targetRadius + breathingEffect;

                    // Adjust orbit speed based on distance (faster when far, slower when close)
                    const distanceFactor = (finalRadius - minRadius) / radiusRange;
                    autoOrbitState.orbitAngle += distanceFactor * 0.003; // Extra rotation when far

                    // Calculate camera position using spherical coordinates
                    cameraReference.value.position.x = finalRadius * Math.sin(autoOrbitState.verticalAngle) * Math.cos(autoOrbitState.orbitAngle);
                    cameraReference.value.position.y = finalRadius * Math.cos(autoOrbitState.verticalAngle) * 0.5;
                    cameraReference.value.position.z = finalRadius * Math.sin(autoOrbitState.verticalAngle) * Math.sin(autoOrbitState.orbitAngle);

                    // Always look at the center (planet) with slight offset for dynamism
                    const lookAtOffset = new Vector3(
                        Math.sin(floatTime * 0.1) * 0.15,
                        Math.cos(floatTime * 0.12) * 0.1,
                        0
                    );
                    cameraReference.value.lookAt(lookAtOffset);
                }
            } else {
                // Sync orbit angle with current camera position when user releases control
                const spherical = new Spherical();
                spherical.setFromVector3(cameraReference.value.position);
                autoOrbitState.orbitAngle = spherical.theta;
                autoOrbitState.verticalAngle = spherical.phi;
            }
        }

        // Update floating animation
        floatTime += 0.016; // ~60fps delta

        if (planetElement.value) {
            planetElement.value.rotation.y += 0.003;
            planetElement.value.rotation.x += 0.001;

            // Subtle floating effect
            const floatY = Math.sin(floatTime * FLOAT_SPEED) * FLOAT_AMPLITUDE;
            const floatX = Math.cos(floatTime * FLOAT_SPEED * 0.7) * FLOAT_AMPLITUDE * 0.5;
            planetElement.value.position.y = floatY;
            planetElement.value.position.x = floatX;
        }

        if (moonsReference.value) {
            moonsReference.value.forEach((moon, index) => {
                moon.orbitAngle += moon.orbitSpeed;

                moon.mesh.position.x = Math.cos(moon.orbitAngle) * moon.orbitRadius;
                moon.mesh.position.z = Math.sin(moon.orbitAngle) * moon.orbitRadius;

                // Individual floating for moons
                const moonFloat = Math.sin(floatTime * FLOAT_SPEED * 1.2 + index) * FLOAT_AMPLITUDE * 0.3;
                moon.mesh.position.y += moonFloat;

                moon.mesh.rotation.y += 0.003;
            });
        }

        // Animate comets
        if (cometsReference.value && scene.value) {
            cometsReference.value.forEach(comet => {
                if (!comet.active) return;

                comet.progress += comet.speed;

                if (comet.progress >= 1) {
                    // Reset comet to a new trajectory
                    comet.progress = 0;
                    comet.active = false;
                    comet.head.visible = false;
                    comet.tail.visible = false;
                } else {
                    // Interpolate position along trajectory
                    const pos = new Vector3().lerpVectors(
                        comet.startPoint,
                        comet.endPoint,
                        comet.progress
                    );
                    comet.head.position.copy(pos);

                    // Update tail geometry
                    const tailPositions = comet.tail.geometry.attributes.position as BufferAttribute;
                    const tailLength = 8;
                    for (let i = 0; i < tailLength; i++) {
                        const t = Math.max(0, comet.progress - (i * 0.015));
                        const tailPos = new Vector3().lerpVectors(
                            comet.startPoint,
                            comet.endPoint,
                            t
                        );
                        tailPositions.setXYZ(i, tailPos.x, tailPos.y, tailPos.z);
                    }
                    tailPositions.needsUpdate = true;
                }
            });

            // Randomly activate inactive comets (higher probability for more action)
            if (Math.random() < 0.008) {
                const inactiveComet = cometsReference.value.find(c => !c.active);
                if (inactiveComet) {
                    // Generate new random trajectory
                    const angle = Math.random() * Math.PI * 2;
                    const startDist = 60 + Math.random() * 40;
                    const endDist = 60 + Math.random() * 40;
                    const yVariation = (Math.random() - 0.5) * 60;

                    inactiveComet.startPoint.set(
                        Math.cos(angle) * startDist,
                        yVariation,
                        Math.sin(angle) * startDist
                    );
                    inactiveComet.endPoint.set(
                        Math.cos(angle + Math.PI + (Math.random() - 0.5)) * endDist,
                        yVariation + (Math.random() - 0.5) * 20,
                        Math.sin(angle + Math.PI + (Math.random() - 0.5)) * endDist
                    );
                    inactiveComet.progress = 0;
                    inactiveComet.speed = 0.002 + Math.random() * 0.003;
                    inactiveComet.active = true;
                    inactiveComet.head.visible = true;
                    inactiveComet.tail.visible = true;
                }
            }
        }

        // Animate twinkling stars
        if (twinklingStarsReference.value) {
            twinklingStarsReference.value.forEach(star => {
                const twinkle = Math.sin(floatTime * star.twinkleSpeed + star.twinkleOffset);
                const scale = star.baseScale * (0.7 + twinkle * 0.3);
                star.sprite.scale.set(scale, scale, 1);

                // Subtle brightness pulse via opacity
                const material = star.sprite.material as SpriteMaterial;
                material.opacity = 0.8 + twinkle * 0.2;
            });
        }

        // Animate asteroids
        if (asteroidsReference.value) {
            asteroidsReference.value.forEach(asteroid => {
                // Rotate asteroid on all axes
                asteroid.mesh.rotation.x += asteroid.rotationSpeed.x;
                asteroid.mesh.rotation.y += asteroid.rotationSpeed.y;
                asteroid.mesh.rotation.z += asteroid.rotationSpeed.z;

                // Orbital movement
                asteroid.orbitAngle += asteroid.orbitSpeed;
                const baseY = Math.sin(asteroid.orbitTilt) * asteroid.orbitRadius * 0.3;
                asteroid.mesh.position.x = Math.cos(asteroid.orbitAngle) * asteroid.orbitRadius;
                asteroid.mesh.position.z = Math.sin(asteroid.orbitAngle) * asteroid.orbitRadius;

                // Subtle floating effect
                const floatY = Math.sin(floatTime * 0.5 + asteroid.floatOffset) * 0.1;
                asteroid.mesh.position.y = baseY + floatY;
            });
        }

        // Animate shooting stars
        if (shootingStarsReference.value && scene.value) {
            shootingStarsReference.value.forEach(star => {
                if (!star.active) return;

                star.progress += star.speed;

                if (star.progress >= 1) {
                    star.active = false;
                    star.line.visible = false;
                } else {
                    // Update line position to create streak effect
                    const currentPos = new Vector3().lerpVectors(
                        star.startPoint,
                        star.endPoint,
                        star.progress
                    );
                    const trailPos = new Vector3().lerpVectors(
                        star.startPoint,
                        star.endPoint,
                        Math.max(0, star.progress - 0.1)
                    );

                    const positions = star.line.geometry.attributes.position as BufferAttribute;
                    positions.setXYZ(0, trailPos.x, trailPos.y, trailPos.z);
                    positions.setXYZ(1, currentPos.x, currentPos.y, currentPos.z);
                    positions.needsUpdate = true;

                    // Fade out as it travels
                    const material = star.line.material as LineBasicMaterial;
                    material.opacity = 1 - star.progress * 0.5;
                }
            });

            // Randomly spawn shooting stars
            if (Math.random() < 0.01) {
                const inactiveStar = shootingStarsReference.value.find(s => !s.active);
                if (inactiveStar) {
                    const angle = Math.random() * Math.PI * 2;
                    const startDist = 30 + Math.random() * 50;
                    const yStart = (Math.random() - 0.5) * 40;

                    inactiveStar.startPoint.set(
                        Math.cos(angle) * startDist,
                        yStart,
                        Math.sin(angle) * startDist
                    );
                    inactiveStar.endPoint.set(
                        Math.cos(angle + 0.3) * (startDist - 20),
                        yStart - 10,
                        Math.sin(angle + 0.3) * (startDist - 20)
                    );
                    inactiveStar.progress = 0;
                    inactiveStar.speed = 0.02 + Math.random() * 0.03;
                    inactiveStar.active = true;
                    inactiveStar.line.visible = true;
                }
            }
        }

        // Animate galaxies (slow rotation)
        if (galaxiesReference.value) {
            galaxiesReference.value.forEach(galaxy => {
                galaxy.sprite.material.rotation += galaxy.rotationSpeed;
            });
        }

        // Animate distant explosions
        if (explosionsReference.value && scene.value) {
            explosionsReference.value.forEach(explosion => {
                if (!explosion.active) return;

                explosion.progress += 0.008;

                if (explosion.progress >= 1) {
                    explosion.active = false;
                    explosion.sprite.visible = false;
                } else {
                    // Expansion effect - fast at first, then slows
                    const easedProgress = 1 - Math.pow(1 - explosion.progress, 3);
                    const currentScale = explosion.maxScale * easedProgress;
                    explosion.sprite.scale.set(currentScale, currentScale, 1);

                    // Color transition: white -> yellow -> orange -> red -> fade
                    const material = explosion.sprite.material as SpriteMaterial;

                    if (explosion.progress < 0.2) {
                        // Bright white flash
                        material.opacity = 1;
                        material.color.setRGB(1, 1, 1);
                    } else if (explosion.progress < 0.4) {
                        // Yellow-orange
                        const t = (explosion.progress - 0.2) / 0.2;
                        material.color.setRGB(1, 1 - t * 0.3, 1 - t * 0.6);
                        material.opacity = 1;
                    } else if (explosion.progress < 0.7) {
                        // Orange-red with original color influence
                        const t = (explosion.progress - 0.4) / 0.3;
                        const r = explosion.color.r / 255;
                        const g = explosion.color.g / 255;
                        const b = explosion.color.b / 255;
                        material.color.setRGB(
                            0.9 + r * 0.1,
                            0.5 * (1 - t) + g * t,
                            0.3 * (1 - t) + b * t
                        );
                        material.opacity = 1 - t * 0.3;
                    } else {
                        // Fade out
                        const t = (explosion.progress - 0.7) / 0.3;
                        material.opacity = 0.7 * (1 - t);
                    }
                }
            });

            // Randomly trigger new explosions (rare event)
            if (Math.random() < 0.002) {
                const inactiveExplosion = explosionsReference.value.find(e => !e.active);
                if (inactiveExplosion) {
                    // Random position in the distance
                    const distance = 100 + Math.random() * 100;
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.random() * Math.PI;

                    inactiveExplosion.position.set(
                        distance * Math.sin(phi) * Math.cos(theta),
                        distance * Math.sin(phi) * Math.sin(theta),
                        distance * Math.cos(phi)
                    );
                    inactiveExplosion.sprite.position.copy(inactiveExplosion.position);
                    inactiveExplosion.progress = 0;
                    inactiveExplosion.maxScale = 8 + Math.random() * 12;
                    inactiveExplosion.active = true;
                    inactiveExplosion.sprite.visible = true;
                    inactiveExplosion.sprite.scale.set(0.1, 0.1, 1);
                }
            }
        }

        if (renderer.value && scene.value && cameraReference.value) {
            renderer.value.render(scene.value, cameraReference.value);
        }
    }

    function init(mountElement: HTMLDivElement | null) {
        if (!mountElement) {
            return;
        }

        const _scene = new Scene();
        scene.value = _scene;

        const _camera = new PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000,
        );

        _camera.position.set(0, 0, 5);
        cameraReference.value = _camera;

        const _renderer = new WebGLRenderer({
            antialias: true,
        });
        _renderer.setSize(window.innerWidth, window.innerHeight);
        _renderer.setClearColor(0x000011, 1);
        _renderer.shadowMap.enabled = true;
        _renderer.shadowMap.type = PCFSoftShadowMap;

        renderer.value = _renderer;
        mountElement.appendChild(_renderer.domElement);

        _renderer.domElement.addEventListener("mousedown", MouseEvent.handleMouseDown);
        _renderer.domElement.addEventListener("mouseup", MouseEvent.handleMouseUp);
        _renderer.domElement.addEventListener("mousemove", MouseEvent.handleMouseMove);
        _renderer.domElement.addEventListener("wheel", MouseEvent.handleMouseWheel, {
            passive: false,
        });

        _renderer.domElement.addEventListener("contextmenu", MouseEvent.handleContextMenu);

        window.addEventListener("keydown", KeyboardEvent.handleKeyDown);
        window.addEventListener("keyup", KeyboardEvent.handleKeyUp);
        window.addEventListener("resize", Window.handleResize);

        const geometry = new SphereGeometry(1.5, 64, 64);

        // Use the new procedural planet texture
        const texture = createPlanetTexture();

        const material = new MeshPhongMaterial({
            map: texture,
            shininess: 30,
            transparent: false,
        });

        const _planet = new Mesh(geometry, material);
        _planet.castShadow = true;
        _planet.receiveShadow = true;
        planetElement.value = _planet;
        _scene.add(_planet);

        // Enhanced ambient light with blue tint for space atmosphere
        const _ambientLight = new AmbientLight(0x1a237e, 0.4);
        _scene.add(_ambientLight);

        // Main directional light - creates dramatic shadows (sun-like)
        const _directionalLight = new DirectionalLight(0xfff4e6, 1.5);
        _directionalLight.position.set(8, 4, 6);
        _directionalLight.castShadow = true;
        _directionalLight.shadow.mapSize.width = 2048;
        _directionalLight.shadow.mapSize.height = 2048;
        _directionalLight.shadow.camera.near = 0.5;
        _directionalLight.shadow.camera.far = 50;
        _scene.add(_directionalLight);

        // Rim light - creates atmospheric glow on the dark side
        const _rimLight = new PointLight(0x4fc3f7, 0.6, 100);
        _rimLight.position.set(-6, 2, -4);
        _scene.add(_rimLight);

        // Secondary fill light for depth
        const _pointLight = new PointLight(0x7c4dff, 0.3, 100);
        _pointLight.position.set(-5, 5, 5);
        _scene.add(_pointLight);

        // === NEBULA SPRITES ===
        // Create nebula background with layered sprites
        createNebulae(_scene);

        const _starsGeometry = new BufferGeometry();

        const positions = new Float32Array(CONSTANTS.HOME.STARS_COUNT * 3);
        const colors = new Float32Array(CONSTANTS.HOME.STARS_COUNT * 3);
        const sizes = new Float32Array(CONSTANTS.HOME.STARS_COUNT);

        for (let i = 0; i < CONSTANTS.HOME.STARS_COUNT; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 400;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 400;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 400;
            sizes[i] = Math.random() * 3 + 0.5;

            // Assign star colors with more variety
            const color = Math.random();

            if (color < 0.5) {
                // White stars (most common)
                colors[i * 3] = 1;
                colors[i * 3 + 1] = 1;
                colors[i * 3 + 2] = 1;
            } else if (color < 0.6) {
                // Blue-white stars
                colors[i * 3] = 0.7;
                colors[i * 3 + 1] = 0.85;
                colors[i * 3 + 2] = 1;
            } else if (color < 0.7) {
                // Yellow-orange stars
                colors[i * 3] = 1;
                colors[i * 3 + 1] = 0.85;
                colors[i * 3 + 2] = 0.6;
            } else if (color < 0.78) {
                // Cyan/teal stars
                colors[i * 3] = 0.5;
                colors[i * 3 + 1] = 1;
                colors[i * 3 + 2] = 0.95;
            } else if (color < 0.84) {
                // Pink stars
                colors[i * 3] = 1;
                colors[i * 3 + 1] = 0.6;
                colors[i * 3 + 2] = 0.8;
            } else if (color < 0.90) {
                // Orange stars
                colors[i * 3] = 1;
                colors[i * 3 + 1] = 0.7;
                colors[i * 3 + 2] = 0.4;
            } else if (color < 0.95) {
                // Purple stars
                colors[i * 3] = 0.8;
                colors[i * 3 + 1] = 0.5;
                colors[i * 3 + 2] = 1;
            } else {
                // Green stars (rare)
                colors[i * 3] = 0.5;
                colors[i * 3 + 1] = 1;
                colors[i * 3 + 2] = 0.6;
            }
        }

        _starsGeometry.setAttribute("position", new BufferAttribute(positions, 3));
        _starsGeometry.setAttribute("color", new BufferAttribute(colors, 3));
        _starsGeometry.setAttribute("size", new BufferAttribute(sizes, 1));

        const starTexture = createStarTexture();
        const starsMaterial = new PointsMaterial({
            size: 2,
            map: starTexture,
            transparent: true,
            alphaTest: 0.001,
            vertexColors: true,
        });

        const stars = new Points(_starsGeometry, starsMaterial);
        _scene.add(stars);

        for (let i = 0; i < CONSTANTS.HOME.MOON_COUNT; i++) {
            const moonSize = 0.2 + Math.random() * 0.3;
            const moonGeometry = new SphereGeometry(moonSize, 32, 32);

            // eslint-disable-next-line style/operator-linebreak
            const colorScheme =
                MOON_COLOR_SCHEMES[Math.floor(Math.random() * MOON_COLOR_SCHEMES.length)]!;

            const moonTexture = createMoonTexture(colorScheme);

            const moonMaterial = new MeshPhongMaterial({
                map: moonTexture,
                shininess: 20,
                transparent: false,
            });

            const moon = new Mesh(moonGeometry, moonMaterial);

            const orbitRadius = 2.5 + i * 1.2;
            const orbitAngle = (i / CONSTANTS.HOME.MOON_COUNT) * Math.PI * 2;
            const orbitSpeed = 0.0003 + Math.random() * 0.0002;

            moon.position.set(
                Math.cos(orbitAngle) * orbitRadius,
                (Math.random() - 0.5) * 0.5,
                Math.sin(orbitAngle) * orbitRadius,
            );

            moon.castShadow = true;
            moon.receiveShadow = true;

            moonsReference.value.push({
                mesh: moon,
                orbitSpeed,
                orbitRadius,
                orbitAngle,
            });

            _scene.add(moon);
        }

        // === COMETS ===
        // Create a pool of comets (inactive initially, activated randomly during animation)
        const cometCount = 6;
        for (let i = 0; i < cometCount; i++) {
            // Comet head - small glowing sphere
            const headGeometry = new SphereGeometry(0.15, 8, 8);
            const headMaterial = new MeshPhongMaterial({
                color: 0xaaddff,
                emissive: 0x4488cc,
                emissiveIntensity: 0.8,
                shininess: 100,
            });
            const head = new Mesh(headGeometry, headMaterial);
            head.visible = false;

            // Comet tail - line with gradient effect
            const tailLength = 8;
            const tailGeometry = new BufferGeometry();
            const tailPositions = new Float32Array(tailLength * 3);
            tailGeometry.setAttribute("position", new BufferAttribute(tailPositions, 3));

            const tailMaterial = new LineBasicMaterial({
                color: 0x88ccff,
                transparent: true,
                opacity: 0.6,
            });
            const tail = new Line(tailGeometry, tailMaterial);
            tail.visible = false;

            _scene.add(head);
            _scene.add(tail);

            cometsReference.value.push({
                head,
                tail,
                progress: 0,
                speed: 0.003,
                startPoint: new Vector3(0, 0, 0),
                endPoint: new Vector3(0, 0, 0),
                active: false,
            });
        }

        // === TWINKLING BRIGHT STARS ===
        // Create special bright stars that twinkle with vibrant colors
        const twinklingStarCount = 25;
        const starColors = [
            0xffffff, // White
            0xaaddff, // Blue-white
            0xffddaa, // Yellow-white
            0xffaaaa, // Red-white
            0xaaffaa, // Green-white
            0xff99cc, // Hot pink
            0xffaa66, // Orange
            0x66ffff, // Cyan
            0xcc99ff, // Light purple
            0xffdd55, // Gold
        ];

        for (let i = 0; i < twinklingStarCount; i++) {
            // Create star texture for sprite
            const canvas = document.createElement("canvas");
            const size = 64;
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext("2d")!;

            // Star glow gradient
            const gradient = ctx.createRadialGradient(
                size / 2, size / 2, 0,
                size / 2, size / 2, size / 2
            );
            gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
            gradient.addColorStop(0.1, "rgba(255, 255, 255, 0.9)");
            gradient.addColorStop(0.3, "rgba(200, 220, 255, 0.5)");
            gradient.addColorStop(0.6, "rgba(100, 150, 255, 0.15)");
            gradient.addColorStop(1, "rgba(50, 100, 200, 0)");

            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, size, size);

            // Add cross flare
            ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(size / 2, 5);
            ctx.lineTo(size / 2, size - 5);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(5, size / 2);
            ctx.lineTo(size - 5, size / 2);
            ctx.stroke();

            const starTexture = new CanvasTexture(canvas);

            const spriteMaterial = new SpriteMaterial({
                map: starTexture,
                color: starColors[Math.floor(Math.random() * starColors.length)],
                transparent: true,
                blending: AdditiveBlending,
                depthWrite: false,
            });

            const sprite = new Sprite(spriteMaterial);

            // Position at far distances
            const distance = 50 + Math.random() * 100;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            sprite.position.x = distance * Math.sin(phi) * Math.cos(theta);
            sprite.position.y = distance * Math.sin(phi) * Math.sin(theta);
            sprite.position.z = distance * Math.cos(phi);

            const baseScale = 2 + Math.random() * 3;
            sprite.scale.set(baseScale, baseScale, 1);

            _scene.add(sprite);

            twinklingStarsReference.value.push({
                sprite,
                baseScale,
                twinkleSpeed: 2 + Math.random() * 4,
                twinkleOffset: Math.random() * Math.PI * 2,
            });
        }

        // === ASTEROIDS ===
        // Create asteroids floating in space with varied shapes
        const asteroidCount = 20;
        const asteroidColors = [
            0x4a4a4a, // Dark grey
            0x6b6b6b, // Medium grey
            0x5c4033, // Brown
            0x8b7355, // Light brown
            0x7c4d3a, // Reddish brown
            0x3d3d3d, // Very dark grey
        ];

        for (let i = 0; i < asteroidCount; i++) {
            // Create irregular asteroid geometry by modifying a sphere
            const baseSize = 0.08 + Math.random() * 0.15;
            const asteroidGeometry = new SphereGeometry(baseSize, 6, 5);

            // Deform vertices to create irregular rocky shape
            const positionAttribute = asteroidGeometry.getAttribute("position");
            const vertex = new Vector3();
            for (let j = 0; j < positionAttribute.count; j++) {
                vertex.fromBufferAttribute(positionAttribute, j);
                const noise = 0.7 + Math.random() * 0.6;
                vertex.multiplyScalar(noise);
                positionAttribute.setXYZ(j, vertex.x, vertex.y, vertex.z);
            }
            asteroidGeometry.computeVertexNormals();

            const asteroidMaterial = new MeshPhongMaterial({
                color: asteroidColors[Math.floor(Math.random() * asteroidColors.length)],
                shininess: 5,
                flatShading: true,
            });

            const asteroid = new Mesh(asteroidGeometry, asteroidMaterial);

            // Position in an asteroid belt-like distribution
            const orbitRadius = 5 + Math.random() * 20;
            const orbitAngle = Math.random() * Math.PI * 2;
            const orbitTilt = (Math.random() - 0.5) * Math.PI * 0.3;

            asteroid.position.x = Math.cos(orbitAngle) * orbitRadius;
            asteroid.position.z = Math.sin(orbitAngle) * orbitRadius;
            asteroid.position.y = Math.sin(orbitTilt) * orbitRadius * 0.3;

            // Random initial rotation
            asteroid.rotation.x = Math.random() * Math.PI * 2;
            asteroid.rotation.y = Math.random() * Math.PI * 2;
            asteroid.rotation.z = Math.random() * Math.PI * 2;

            asteroid.castShadow = true;
            asteroid.receiveShadow = true;

            _scene.add(asteroid);

            asteroidsReference.value.push({
                mesh: asteroid,
                rotationSpeed: {
                    x: (Math.random() - 0.5) * 0.02,
                    y: (Math.random() - 0.5) * 0.02,
                    z: (Math.random() - 0.5) * 0.01,
                },
                orbitSpeed: 0.0001 + Math.random() * 0.0003,
                orbitRadius,
                orbitAngle,
                orbitTilt,
                floatOffset: Math.random() * Math.PI * 2,
            });
        }

        // === SHOOTING STARS ===
        // Create a pool of shooting stars
        const shootingStarCount = 5;
        for (let i = 0; i < shootingStarCount; i++) {
            const geometry = new BufferGeometry();
            const positions = new Float32Array(6); // 2 points, 3 coords each
            geometry.setAttribute("position", new BufferAttribute(positions, 3));

            const material = new LineBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 1,
                linewidth: 2,
            });

            const line = new Line(geometry, material);
            line.visible = false;

            _scene.add(line);

            shootingStarsReference.value.push({
                line,
                progress: 0,
                speed: 0.03,
                startPoint: new Vector3(0, 0, 0),
                endPoint: new Vector3(0, 0, 0),
                active: false,
            });
        }

        // === DISTANT GALAXIES ===
        // Create spiral galaxy sprites in the far background
        const galaxyCount = 6;
        const galaxyColors = [
            { r: 255, g: 200, b: 150 },  // Warm spiral
            { r: 150, g: 180, b: 255 },  // Blue spiral
            { r: 255, g: 150, b: 200 },  // Pink spiral
            { r: 200, g: 255, b: 200 },  // Green tinted
            { r: 255, g: 220, b: 180 },  // Golden
            { r: 180, g: 150, b: 255 },  // Purple
        ];

        for (let i = 0; i < galaxyCount; i++) {
            const canvas = document.createElement("canvas");
            const size = 128;
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext("2d")!;

            const galaxyColor = galaxyColors[i % galaxyColors.length]!;

            // Draw spiral galaxy
            const centerX = size / 2;
            const centerY = size / 2;

            // Central bright core
            const coreGradient = ctx.createRadialGradient(
                centerX, centerY, 0,
                centerX, centerY, size * 0.15
            );
            coreGradient.addColorStop(0, `rgba(${galaxyColor.r}, ${galaxyColor.g}, ${galaxyColor.b}, 0.9)`);
            coreGradient.addColorStop(0.5, `rgba(${galaxyColor.r}, ${galaxyColor.g}, ${galaxyColor.b}, 0.4)`);
            coreGradient.addColorStop(1, "rgba(0, 0, 0, 0)");
            ctx.fillStyle = coreGradient;
            ctx.fillRect(0, 0, size, size);

            // Spiral arms
            ctx.strokeStyle = `rgba(${galaxyColor.r}, ${galaxyColor.g}, ${galaxyColor.b}, 0.3)`;
            ctx.lineWidth = 3;

            for (let arm = 0; arm < 2; arm++) {
                ctx.beginPath();
                const armOffset = arm * Math.PI;
                for (let angle = 0; angle < Math.PI * 4; angle += 0.1) {
                    const r = 5 + angle * 6;
                    const x = centerX + Math.cos(angle + armOffset) * r;
                    const y = centerY + Math.sin(angle + armOffset) * r;
                    if (angle === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }

            // Add star clusters along arms
            for (let j = 0; j < 30; j++) {
                const angle = Math.random() * Math.PI * 4;
                const r = 10 + angle * 5 + (Math.random() - 0.5) * 10;
                const x = centerX + Math.cos(angle) * r;
                const y = centerY + Math.sin(angle) * r;
                const starSize = Math.random() * 2 + 0.5;
                const alpha = 0.3 + Math.random() * 0.4;

                ctx.fillStyle = `rgba(${galaxyColor.r}, ${galaxyColor.g}, ${galaxyColor.b}, ${alpha})`;
                ctx.beginPath();
                ctx.arc(x, y, starSize, 0, Math.PI * 2);
                ctx.fill();
            }

            const galaxyTexture = new CanvasTexture(canvas);

            const spriteMaterial = new SpriteMaterial({
                map: galaxyTexture,
                transparent: true,
                blending: AdditiveBlending,
                depthWrite: false,
            });

            const sprite = new Sprite(spriteMaterial);

            // Position very far in the background
            const distance = 180 + Math.random() * 70;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            sprite.position.x = distance * Math.sin(phi) * Math.cos(theta);
            sprite.position.y = distance * Math.sin(phi) * Math.sin(theta);
            sprite.position.z = distance * Math.cos(phi);

            const scale = 15 + Math.random() * 20;
            sprite.scale.set(scale, scale, 1);

            // Random initial rotation
            sprite.material.rotation = Math.random() * Math.PI * 2;

            _scene.add(sprite);

            galaxiesReference.value.push({
                sprite,
                rotationSpeed: (Math.random() - 0.5) * 0.0005,
            });
        }

        // === DISTANT EXPLOSIONS (SUPERNOVAS) ===
        // Create a pool of explosion sprites
        const explosionCount = 4;
        const explosionColors = [
            { r: 255, g: 150, b: 50 },   // Orange
            { r: 50, g: 200, b: 255 },   // Cyan
            { r: 255, g: 100, b: 200 },  // Magenta
            { r: 255, g: 220, b: 100 },  // Yellow
        ];

        for (let i = 0; i < explosionCount; i++) {
            const canvas = document.createElement("canvas");
            const size = 128;
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext("2d")!;

            // Create starburst explosion texture
            const centerX = size / 2;
            const centerY = size / 2;

            // Radial gradient for core glow
            const coreGradient = ctx.createRadialGradient(
                centerX, centerY, 0,
                centerX, centerY, size / 2
            );
            coreGradient.addColorStop(0, "rgba(255, 255, 255, 1)");
            coreGradient.addColorStop(0.1, "rgba(255, 255, 200, 0.9)");
            coreGradient.addColorStop(0.3, "rgba(255, 200, 100, 0.6)");
            coreGradient.addColorStop(0.5, "rgba(255, 150, 50, 0.3)");
            coreGradient.addColorStop(0.7, "rgba(255, 100, 50, 0.1)");
            coreGradient.addColorStop(1, "rgba(0, 0, 0, 0)");

            ctx.fillStyle = coreGradient;
            ctx.fillRect(0, 0, size, size);

            // Add rays/spikes
            ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
            ctx.lineWidth = 2;
            const rayCount = 12;
            for (let j = 0; j < rayCount; j++) {
                const angle = (j / rayCount) * Math.PI * 2;
                const innerRadius = size * 0.1;
                const outerRadius = size * 0.45 + Math.random() * size * 0.1;

                ctx.beginPath();
                ctx.moveTo(
                    centerX + Math.cos(angle) * innerRadius,
                    centerY + Math.sin(angle) * innerRadius
                );
                ctx.lineTo(
                    centerX + Math.cos(angle) * outerRadius,
                    centerY + Math.sin(angle) * outerRadius
                );
                ctx.stroke();
            }

            // Add scattered particles
            for (let j = 0; j < 20; j++) {
                const angle = Math.random() * Math.PI * 2;
                const dist = Math.random() * size * 0.4;
                const px = centerX + Math.cos(angle) * dist;
                const py = centerY + Math.sin(angle) * dist;
                const particleSize = Math.random() * 3 + 1;

                ctx.fillStyle = `rgba(255, 255, 200, ${0.3 + Math.random() * 0.4})`;
                ctx.beginPath();
                ctx.arc(px, py, particleSize, 0, Math.PI * 2);
                ctx.fill();
            }

            const explosionTexture = new CanvasTexture(canvas);
            const explosionColor = explosionColors[i % explosionColors.length]!;

            const spriteMaterial = new SpriteMaterial({
                map: explosionTexture,
                transparent: true,
                blending: AdditiveBlending,
                depthWrite: false,
                opacity: 0,
            });

            const sprite = new Sprite(spriteMaterial);
            sprite.visible = false;
            sprite.scale.set(0.1, 0.1, 1);

            _scene.add(sprite);

            explosionsReference.value.push({
                sprite,
                progress: 0,
                maxScale: 10,
                position: new Vector3(0, 0, 0),
                color: explosionColor,
                active: false,
            });
        }

        animate();
    }

    function cleanup(mountElement: HTMLDivElement | null) {
        if (animationIdReference.value) {
            cancelAnimationFrame(animationIdReference.value);
        }

        window.removeEventListener("resize", Window.handleResize);
        window.removeEventListener("keydown", KeyboardEvent.handleKeyDown);
        window.removeEventListener("keyup", KeyboardEvent.handleKeyUp);

        if (renderer.value) {
            renderer.value.domElement.removeEventListener("mousedown", MouseEvent.handleMouseDown);
            renderer.value.domElement.removeEventListener("mouseup", MouseEvent.handleMouseUp);
            renderer.value.domElement.removeEventListener("mousemove", MouseEvent.handleMouseMove);
            renderer.value.domElement.removeEventListener("wheel", MouseEvent.handleMouseWheel);
            renderer.value.domElement.removeEventListener(
                "contextmenu",
                MouseEvent.handleContextMenu,
            );

            if (mountElement && renderer.value.domElement.parentNode) {
                mountElement.removeChild(renderer.value.domElement);
            }

            renderer.value.dispose();
        }
    }

    return {
        renderer,
        scene,
        cameraReference,
        planetElement,
        animationIdReference,
        MouseEvent,
        KeyboardEvent,
        Camera,
        Window,
        mouseReference,
        keysReference,
        animate,
        init,
        cleanup,
    };
}

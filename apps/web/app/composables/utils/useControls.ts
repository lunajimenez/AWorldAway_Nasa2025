import {
    AmbientLight,
    BufferAttribute,
    BufferGeometry,
    CanvasTexture,
    DirectionalLight,
    Mesh,
    MeshPhongMaterial,
    PCFSoftShadowMap,
    PerspectiveCamera,
    PointLight,
    Points,
    PointsMaterial,
    RepeatWrapping,
    Scene,
    SphereGeometry,
    Spherical,
    Vector3,
    WebGLRenderer,
} from "three";

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

    const MouseEvent = {
        handleMouseDown(event: MouseEvent) {
            mouseReference.meta.isDown = true;
            mouseReference.x = event.clientX;
            mouseReference.y = event.clientY;
        },
        handleMouseUp(_event: MouseEvent) {
            mouseReference.meta.isDown = false;
        },
        handleMouseMove(event: MouseEvent) {
            if (!mouseReference.meta.isDown || !cameraReference.value) {
                return;
            }

            const deltaX = event.clientX - mouseReference.x;
            const deltaY = event.clientY - mouseReference.y;

            const spherical = new Spherical();
            spherical.setFromVector3(cameraReference.value.position);

            spherical.theta -= deltaX * 0.01;
            spherical.phi += deltaY * 0.01;
            spherical.phi = Math.max(0.1, 1, Math.min(Math.PI - 0.1, spherical.phi));

            cameraReference.value.position.setFromSpherical(spherical);
            cameraReference.value.lookAt(0, 0, 0);

            mouseReference.x = event.clientX;
            mouseReference.y = event.clientY;
        },
        handleMouseWheel(event: WheelEvent) {
            if (!cameraReference.value) {
                return;
            }

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

        if (planetElement.value) {
            planetElement.value.rotation.y += 0.005;
            planetElement.value.rotation.x += 0.002;
        }

        if (moonsReference.value) {
            moonsReference.value.forEach((moon) => {
                moon.orbitAngle += moon.orbitSpeed;

                moon.mesh.position.x = Math.cos(moon.orbitAngle) * moon.orbitRadius;
                moon.mesh.position.z = Math.sin(moon.orbitAngle) * moon.orbitRadius;

                moon.mesh.rotation.y += 0.003;
            });
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

        const canvas = document.createElement("canvas");
        canvas.width = 512;
        canvas.height = 512;

        const context = canvas.getContext("2d")!;

        const gradient = context.createLinearGradient(0, 0, 512, 512);
        gradient.addColorStop(0, "#1e40af");
        gradient.addColorStop(0.3, "#3b82f6");
        gradient.addColorStop(0.6, "#1e3a8a");
        gradient.addColorStop(1, "#1e293b");

        context.fillStyle = gradient;
        context.fillRect(0, 0, 512, 512);

        context.fillStyle = "#059669";
        context.beginPath();
        context.arc(150, 150, 80, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = "#16a34a";
        context.beginPath();
        context.arc(350, 200, 60, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = "#854d0e";
        context.beginPath();
        context.arc(250, 350, 90, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = "#166534";
        context.beginPath();
        context.arc(100, 300, 40, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = "#7c2d12";
        context.beginPath();
        context.arc(400, 100, 35, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = "rgba(255, 255, 255, 0.4)";
        for (let i = 0; i < 15; i++) {
            const x = Math.random() * 512;
            const y = Math.random() * 512;
            const radius = Math.random() * 25 + 15;
            context.beginPath();
            context.arc(x, y, radius, 0, Math.PI * 2);
            context.fill();
        }

        context.fillStyle = "rgba(255, 255, 255, 0.8)";

        context.beginPath();
        context.arc(256, 50, 40, 0, Math.PI * 2);
        context.fill();

        context.beginPath();
        context.arc(256, 462, 35, 0, Math.PI * 2);
        context.fill();

        const texture = new CanvasTexture(canvas);
        texture.wrapS = RepeatWrapping;
        texture.wrapT = RepeatWrapping;

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

        const _ambientLight = new AmbientLight(0x404040, 0.3);
        _scene.add(_ambientLight);

        // eslint-disable-next-line unicorn/number-literal-case
        const _directionalLight = new DirectionalLight(0xffffff, 1.2);
        _directionalLight.position.set(5, 3, 5);
        _directionalLight.castShadow = true;
        _directionalLight.shadow.mapSize.width = 2048;
        _directionalLight.shadow.mapSize.height = 2048;
        _scene.add(_directionalLight);

        // eslint-disable-next-line unicorn/number-literal-case
        const _pointLight = new PointLight(0xffffff, 0.4, 100);
        _pointLight.position.set(-5, 5, 5);
        _scene.add(_pointLight);

        const _starsGeometry = new BufferGeometry();

        const positions = new Float32Array(CONSTANTS.HOME.STARS_COUNT * 3);
        const colors = new Float32Array(CONSTANTS.HOME.STARS_COUNT * 3);
        const sizes = new Float32Array(CONSTANTS.HOME.STARS_COUNT);

        for (let i = 0; i < CONSTANTS.HOME.STARS_COUNT; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 400;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 400;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 400;
            sizes[i] = Math.random() * 3 + 0.5;

            const color = Math.random();

            if (color < 0.7) {
                colors[i * 3] = 1;
                colors[i * 3 + 1] = 1;
                colors[i * 3 + 2] = 1;
                continue;
            }

            if (color < 0.85) {
                colors[i * 3] = 0.7;
                colors[i * 3 + 1] = 0.8;
                colors[i * 3 + 2] = 1;
                continue;
            }

            colors[i * 3] = 1;
            colors[i * 3 + 1] = 0.9;
            colors[i * 3 + 2] = 0.7;
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

            animate();
        }
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

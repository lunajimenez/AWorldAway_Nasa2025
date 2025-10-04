<script setup lang="ts">
    import { Settings } from "lucide-vue-next";
    import {
        PerspectiveCamera,
        Spherical,
        Vector3,
        Mesh,
        WebGLRenderer,
        Scene,
        PCFSoftShadowMap,
        SphereGeometry,
        CanvasTexture,
        MeshPhongMaterial,
        AmbientLight,
        RepeatWrapping,
        DirectionalLight,
        PointLight,
        BufferGeometry,
        BufferAttribute,
        PointsMaterial,
        Points,
    } from "three";

    definePageMeta({
        title: "pages.home.title",
    });

    const modal = useModal();

    const mountReference = useTemplateRef("MountRef");

    const renderer = shallowRef<WebGLRenderer>();
    const scene = shallowRef<Scene>();
    const cameraReference = shallowRef<PerspectiveCamera>();
    const planetElement = shallowRef<Mesh>();

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
            if (!mouseReference.meta.isDown || !cameraReference.value) return;

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
            if (!cameraReference.value) return;

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

                case "KeysS": {
                    keysReference.s = true;
                    break;
                }

                case "KeysD": {
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

                case "KeysS": {
                    keysReference.s = false;
                    break;
                }

                case "KeysD": {
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
            if (!cameraReference.value) return;

            const moveSpeed = 0.1;
            const fastSpeed = moveSpeed * 2;

            const forward = new Vector3();
            cameraReference.value.getWorldDirection(forward);

            const right = new Vector3();
            right.crossVectors(forward, cameraReference.value.up).normalize();

            const up = cameraReference.value.up.clone();

            const movement = new Vector3();

            if (keysReference.w) movement.add(forward);
            if (keysReference.s) movement.sub(forward);
            if (keysReference.a) movement.sub(right);
            if (keysReference.d) movement.add(right);
            if (keysReference.space) movement.add(up);
            if (keysReference.shift) movement.sub(up);

            const speed = keysReference.shift && !keysReference.space ? fastSpeed : moveSpeed;
            movement.multiplyScalar(speed);

            cameraReference.value.position.add(movement);
        },
    };

    const Window = {
        handleResize() {
            if (!cameraReference.value || !renderer.value) return;

            cameraReference.value.aspect = window.innerWidth / window.innerHeight;
            cameraReference.value.updateProjectionMatrix();
            renderer.value.setSize(window.innerWidth, window.innerHeight);
        },
    };

    function animate() {
        animationIdReference.value = requestAnimationFrame(animate);

        Camera.update();

        if (planetElement.value) {
            planetElement.value.rotation.y += 0.008;
            planetElement.value.rotation.x += 0.003;
        }

        if (renderer.value && scene.value && cameraReference.value) {
            renderer.value.render(scene.value, cameraReference.value);
        }
    }

    function init() {
        if (!mountReference.value) return;

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

        const _renderer = new WebGLRenderer({ antialias: true });
        _renderer.setSize(window.innerWidth, window.innerHeight);
        _renderer.setClearColor(0x000011, 1);
        _renderer.shadowMap.enabled = true;
        _renderer.shadowMap.type = PCFSoftShadowMap;

        renderer.value = _renderer;
        mountReference.value.appendChild(_renderer.domElement);

        _renderer.domElement.addEventListener("mousedown", MouseEvent.handleMouseDown);
        _renderer.domElement.addEventListener("mouseup", MouseEvent.handleMouseUp);
        _renderer.domElement.addEventListener("mousemove", MouseEvent.handleMouseMove);
        _renderer.domElement.addEventListener("wheel", MouseEvent.handleMouseWheel);

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

        const _directionalLight = new DirectionalLight(0xffffff, 1.2);
        _directionalLight.position.set(5, 3, 5);
        _directionalLight.castShadow = true;
        _directionalLight.shadow.mapSize.width = 2048;
        _directionalLight.shadow.mapSize.height = 2048;
        _scene.add(_directionalLight);

        const _pointLight = new PointLight(0xffffff, 0.4, 100);
        _pointLight.position.set(-5, 5, 5);
        _scene.add(_pointLight);

        const _starsGeometry = new BufferGeometry();
        const STARS_COUNT = 1000;
        const positions = new Float32Array(STARS_COUNT * 3);

        for (let i = 0; i < STARS_COUNT * 3; i++) {
            positions[i] = (Math.random() - 0.5) * 200;
        }

        _starsGeometry.setAttribute("position", new BufferAttribute(positions, 3));
        const starsMaterial = new PointsMaterial({ color: 0xffffff, size: 0.5 });
        const stars = new Points(_starsGeometry, starsMaterial);
        _scene.add(stars);

        animate();
    }

    function cleanup() {
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

            if (mountReference.value && renderer.value.domElement.parentNode) {
                mountReference.value.removeChild(renderer.value.domElement);
            }

            renderer.value.dispose();
        }
    }

    onMounted(() => {
        nextTick(() => {
            init();
        });
    });

    onUnmounted(() => {
        cleanup();
    });
</script>

<template>
    <main class="relative w-full h-screen overflow-hidden">
        <div ref="MountRef" class="w-full h-full" />

        <div
            class="absolute top-0 left-0 z-10 h-full w-full grid grid-cols-12 grid-rows-12 gap-4 p-4 pointer-events-none"
        >
            <div
                class="col-start-1 col-span-8 row-start-1 row-span-2 flex flex-col justify-center pointer-events-auto"
            >
                <h1 class="text-2xl font-bold mb-2">{{ $t("pages.home.ui.title") }}</h1>
                <p class="text-sm opacity-80">{{ $t("pages.home.ui.subtitle") }}</p>
            </div>

            <div
                class="col-start-9 col-span-4 row-start-1 row-span-2 pointer-events-auto flex items-start justify-end"
            >
                <Button
                    @click="
                        () => {
                            modal.loadComponent({
                                loader: () =>
                                    import('@/components/common/settings/CommonSettingsModal.vue'),
                                key: 'settings:modal',
                            });

                            modal.open.value = true;
                        }
                    "
                >
                    <Settings />
                </Button>
            </div>

            <div
                class="col-start-9 col-span-4 row-start-11 row-span-2 flex items-end justify-end pointer-events-auto"
            >
                <div class="p-4 text-xs space-y-1 text-right">
                    <p>{{ $t("pages.home.controls.shift") }} 🚀</p>
                    <p>{{ $t("pages.home.controls.space") }} 🔼</p>
                    <p>{{ $t("pages.home.controls.planet") }} 🌍</p>
                </div>
            </div>
        </div>
    </main>
</template>

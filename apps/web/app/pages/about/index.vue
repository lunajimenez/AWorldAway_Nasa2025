<script setup lang="ts">
    import { ChevronLeft, ChevronRight, ExternalLink, Settings } from "lucide-vue-next";

    definePageMeta({
        title: "pages.about.title",
    });

    interface TeamMember {
        id: number;
        name: string;
        nickname: string;
        roles: string[];
        url: string;
        image: string;
        description: string[];
    }

    const modal = useModal();

    const teamMembers: TeamMember[] = [
        {
            id: 1,
            name: "Luna Jimenez",
            nickname: "Lunita",
            roles: ["pages.about.team.roles.team_leader", "pages.about.team.roles.data_scientist"],
            url: "https://www.linkedin.com/in/luna-katalina-quintero-jimenez/",
            image: "https://media.licdn.com/dms/image/v2/D5603AQGcTMaVghQq1w/profile-displayphoto-shrink_200_200/B56ZSwQwXzGoAY-/0/1738123971155?e=1762387200&v=beta&t=WfEMLUVqAWzTZA1QHX1hyHSmI2Tj1MOoc6_QaHSBJtM",
            description: [
                "pages.about.team.members.luna.bio.p1",
                "pages.about.team.members.luna.bio.p2",
                "pages.about.team.members.luna.bio.p3",
            ],
        },
        {
            id: 2,
            name: "Maria Camila Garcia",
            nickname: "Cami",
            roles: ["pages.about.team.roles.data_scientist"],
            url: "https://www.linkedin.com/in/maria-camila-garc%C3%ADa-salazar-469362241/",
            image: "https://media.licdn.com/dms/image/v2/D4E03AQHkwAssLRrCmQ/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1731373466176?e=1762387200&v=beta&t=No_k4s4CTfi7nCDQAbqy50YF6KfC9-uauX5j4ksf_og",
            description: [
                "pages.about.team.members.cami.bio.p1",
                "pages.about.team.members.cami.bio.p2",
                "pages.about.team.members.cami.bio.p3",
            ],
        },
        {
            id: 3,
            name: "Leonardo Gonzalez",
            nickname: "Leo",
            roles: ["pages.about.team.roles.frontend_developer"],
            url: "https://www.linkedin.com/in/leonardo-gonzalez-castillo-a94098256/",
            image: "https://media.licdn.com/dms/image/v2/D4E03AQG4samTV8NNOg/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1731382900691?e=1762387200&v=beta&t=ytjg6wlxVAP2Ku5Mr-oEWo7lCSFKqKSM8LeLkb_RPzU",
            description: [
                "pages.about.team.members.leo.bio.p1",
                "pages.about.team.members.leo.bio.p2",
                "pages.about.team.members.leo.bio.p3",
            ],
        },
        {
            id: 4,
            name: "Michael Taboada",
            nickname: "Maic",
            roles: ["pages.about.team.roles.data_scientist"],
            url: "https://www.linkedin.com/in/michael-andr%C3%A9s-taboada-naranjo-0263171b1/",
            image: "https://media.licdn.com/dms/image/v2/D5635AQHjKqOpGIlO6A/profile-framedphoto-shrink_200_200/profile-framedphoto-shrink_200_200/0/1713925488966?e=1760295600&v=beta&t=aavL6ghnxDdjr9zDeq0G5FnWqVHHp1Pqyxx70PJvTGE",
            description: [
                "pages.about.team.members.maic.bio.p1",
                "pages.about.team.members.maic.bio.p2",
                "pages.about.team.members.maic.bio.p3",
            ],
        },
        {
            id: 5,
            name: "Mauro Gonzalez",
            nickname: "Mau",
            roles: [
                "pages.about.team.roles.fullstack_developer",
                "pages.about.team.roles.data_scientist",
            ],
            url: "https://www.linkedin.com/in/mauro-gonzalez-figueroa/",
            image: "https://media.licdn.com/dms/image/v2/D4E03AQF-9kGeaEA8zg/profile-displayphoto-scale_200_200/B4EZmMIMO4KQAY-/0/1758992583063?e=1762387200&v=beta&t=bmbw8AAHnhNQYKvNx1fVjuImv2AdOUs2Kn0tXZEPs-I",
            description: [
                "pages.about.team.members.mau.bio.p1",
                "pages.about.team.members.mau.bio.p2",
                "pages.about.team.members.mau.bio.p3",
            ],
        },
    ];

    const currentIndex = ref(0);
    const isTransitioning = ref(false);

    const currentMember = computed(() => teamMembers[currentIndex.value]);

    const goToPrevious = () => {
        if (isTransitioning.value) return;
        isTransitioning.value = true;
        setTimeout(() => {
            currentIndex.value =
                currentIndex.value === 0 ? teamMembers.length - 1 : currentIndex.value - 1;
            isTransitioning.value = false;
        }, 150);
    };

    const goToNext = () => {
        if (isTransitioning.value) return;
        isTransitioning.value = true;
        setTimeout(() => {
            currentIndex.value =
                currentIndex.value === teamMembers.length - 1 ? 0 : currentIndex.value + 1;
            isTransitioning.value = false;
        }, 150);
    };

    const goToSlide = (index: number) => {
        if (isTransitioning.value || index === currentIndex.value) return;
        isTransitioning.value = true;
        setTimeout(() => {
            currentIndex.value = index;
            isTransitioning.value = false;
        }, 150);
    };

    const stars = Array.from({ length: CONSTANTS.HOME.STARS_COUNT }).map(() => ({
        width: Math.random() * 2 + 1,
        height: Math.random() * 2 + 1,
        top: Math.random() * 100,
        left: Math.random() * 100,
        opacity: Math.random() * 0.5 + 0.3,
    }));
</script>

<template>
    <main class="min-h-screen bg-black text-white relative overflow-hidden">
        <!-- Starfield background -->
        <div class="absolute inset-0 overflow-hidden pointer-events-none">
            <div
                v-for="(star, i) in stars"
                :key="i"
                class="absolute bg-white rounded-full"
                :style="{
                    width: `${star.width}px`,
                    height: `${star.height}px`,
                    top: `${star.top}%`,
                    left: `${star.left}%`,
                    opacity: star.opacity,
                }"
            />
        </div>

        <!-- Navigation -->
        <nav class="relative z-10 p-4 flex justify-between items-center">
            <NuxtLink
                :to="$localeRoute({ path: '/' })"
                class="inline-flex items-center gap-2 text-white/70 hover:text-white transition-colors"
            >
                <ChevronLeft :size="20" />
                {{ $t("pages.about.navigation.back") }}
            </NuxtLink>

            <Button
                @click="
                    () => {
                        modal.loadComponent({
                            loader: () =>
                                import(
                                    '@/components/common/settings/modal/CommonSettingsModal.vue'
                                ),
                            key: 'settings:modal',
                        });
                        modal.open.value = true;
                    }
                "
                variant="ghost"
                size="icon"
            >
                <Settings :size="20" />
            </Button>
        </nav>

        <!-- Content -->
        <div class="relative z-10 max-w-6xl mx-auto px-6 py-12">
            <!-- Header -->
            <div class="text-center mb-16">
                <h1
                    class="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent"
                >
                    {{ $t("pages.about.header.title") }}
                </h1>
                <p class="text-xl text-white/70 max-w-2xl mx-auto leading-relaxed">
                    {{ $t("pages.about.header.description") }}
                </p>
            </div>

            <!-- Team Carousel Section -->
            <section class="mb-12">
                <h2
                    class="text-3xl font-bold mb-8 text-center flex items-center justify-center gap-3"
                >
                    <svg class="w-8 h-8" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path
                            d="M12 1.75a5.25 5.25 0 1 1-5.25 5.25A5.26 5.26 0 0 1 12 1.75zm0 9A3.75 3.75 0 1 0 8.25 7 3.75 3.75 0 0 0 12 10.75Zm8 9.5a.75.75 0 0 1-1.06 0 7.93 7.93 0 0 0-5.64-2.34h-2.6A7.93 7.93 0 0 0 5.06 20.25a.75.75 0 0 1-1.06-1.06 9.43 9.43 0 0 1 6.69-2.79h2.6a9.43 9.43 0 0 1 6.69 2.79.75.75 0 0 1 0 1.06Z"
                        />
                        <path
                            d="M21.53 9.22a.75.75 0 0 0-1.06 0L18 11.69l-1.47-1.47a.75.75 0 0 0-1.06 1.06l2 2a.75.75 0 0 0 1.06 0l3-3a.75.75 0 0 0 0-1.06Z"
                        />
                    </svg>
                    {{ $t("pages.about.team.title") }}
                </h2>

                <div v-if="currentMember" class="carousel-container">
                    <article :class="['about-wrapper', { transitioning: isTransitioning }]">
                        <!-- Text Content -->
                        <div class="about-text">
                            <div class="member-header">
                                <h3 class="member-name">
                                    {{ currentMember.name }}
                                    <span class="member-nickname"
                                        >"{{ currentMember.nickname }}"</span
                                    >
                                </h3>
                                <div class="member-roles">
                                    <span v-for="(role, idx) in currentMember.roles" :key="role">
                                        {{ $t(role) }}
                                        <span v-if="idx < currentMember.roles.length - 1"> · </span>
                                    </span>
                                </div>
                            </div>

                            <p
                                v-for="(paragraph, index) in currentMember.description"
                                :key="index"
                                class="mb-4 leading-relaxed"
                            >
                                {{ $t(paragraph) }}
                            </p>

                            <NuxtLink
                                :external="true"
                                :href="currentMember.url"
                                target="_blank"
                                rel="noopener noreferrer"
                                class="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors mt-2"
                            >
                                <ExternalLink :size="18" />
                                <span>LinkedIn</span>
                            </NuxtLink>
                        </div>

                        <!-- Photo -->
                        <div class="about-photo">
                            <figure class="photo-frame">
                                <img
                                    :src="currentMember.image"
                                    :alt="currentMember.name"
                                    loading="lazy"
                                />
                            </figure>
                        </div>
                    </article>

                    <!-- Carousel Controls -->
                    <div class="carousel-controls">
                        <button
                            class="carousel-button prev"
                            @click="goToPrevious"
                            :aria-label="$t('pages.about.carousel.previous')"
                        >
                            <ChevronLeft :size="24" />
                        </button>

                        <div class="carousel-dots">
                            <button
                                v-for="(member, index) in teamMembers"
                                :key="member.id"
                                :class="['dot', { active: index === currentIndex }]"
                                @click="() => goToSlide(index)"
                                :aria-label="`${$t('pages.about.carousel.goTo')} ${member.name}`"
                            />
                        </div>

                        <button
                            class="carousel-button next"
                            @click="goToNext"
                            :aria-label="$t('pages.about.carousel.next')"
                        >
                            <ChevronRight :size="24" />
                        </button>
                    </div>
                </div>
            </section>
        </div>
    </main>
</template>

<style scoped>
    /* Carousel Container */
    .carousel-container {
        position: relative;
    }

    /* Member Card Layout */
    .about-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 2rem;
        opacity: 1;
        transition: opacity 0.3s ease-in-out;
        min-height: 400px;
    }

    .about-wrapper.transitioning {
        opacity: 0;
    }

    @media (min-width: 768px) {
        .about-wrapper {
            flex-direction: row;
        }
    }

    /* Member Header */
    .member-header {
        margin-bottom: 1rem;
    }

    .member-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }

    .member-nickname {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.7;
        margin-left: 0.5rem;
    }

    .member-roles {
        font-size: 1rem;
        color: rgb(96 165 250);
        font-weight: 600;
    }

    /* Text Content */
    .about-text {
        order: 2;
        text-wrap: pretty;
        max-width: 60ch;
    }

    @media (min-width: 768px) {
        .about-text {
            order: 1;
        }
    }

    /* Photo */
    .about-photo {
        order: 1;
        display: grid;
        place-items: center;
    }

    @media (min-width: 768px) {
        .about-photo {
            order: 2;
        }
    }

    .photo-frame {
        width: 16rem;
        aspect-ratio: 1 / 1;
        display: grid;
        place-items: center;
        padding: 0.25rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 1rem;
        transform: rotate(3deg);
        border: 1px solid rgba(0, 0, 0, 0.7);
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.05), 0 10px 15px -3px rgba(0, 0, 0, 0.1),
            0 4px 6px -4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .photo-frame:hover {
        transform: rotate(0deg) scale(1.02);
    }

    @media (min-width: 1024px) {
        .photo-frame {
            padding: 0.5rem;
        }
    }

    .photo-frame img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: 50% 50%;
        border-radius: calc(1rem - 0.25rem);
    }

    /* Carousel Controls */
    .carousel-controls {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-top: 2rem;
    }

    .carousel-button {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border: none;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
    }

    .carousel-button:hover {
        background: rgb(96 165 250);
        color: white;
        transform: scale(1.1);
    }

    .carousel-button:active {
        transform: scale(0.95);
    }

    /* Dots */
    .carousel-dots {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }

    .dot {
        width: 10px;
        height: 10px;
        border: none;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .dot:hover {
        background: rgba(255, 255, 255, 0.4);
        transform: scale(1.2);
    }

    .dot.active {
        width: 24px;
        border-radius: 12px;
        background: rgb(96 165 250);
    }

    /* Mobile Adjustments */
    @media (max-width: 767px) {
        .carousel-controls {
            position: relative;
            padding: 0 1rem;
        }

        .carousel-button {
            width: 35px;
            height: 35px;
        }
    }

    /* Accessibility */
    .carousel-button:focus-visible,
    .dot:focus-visible {
        outline: 2px solid rgb(96 165 250);
        outline-offset: 2px;
    }
</style>

<script setup lang="ts">
    import { ExternalLink, Settings } from "lucide-vue-next";

    definePageMeta({
        title: "pages.about.title",
    });

    interface TeamMember {
        name: string;
        roles: string[];
        url: string;
        image: string;
    }

    const modal = useModal();

    const teamMembers: TeamMember[] = [
        {
            name: "Luna Jimenez",
            roles: ["pages.about.team.roles.team_leader", "pages.about.team.roles.data_scientist"],
            url: "https://www.linkedin.com/in/luna-katalina-quintero-jimenez/",
            image: "https://media.licdn.com/dms/image/v2/D5603AQGcTMaVghQq1w/profile-displayphoto-shrink_200_200/B56ZSwQwXzGoAY-/0/1738123971155?e=1762387200&v=beta&t=WfEMLUVqAWzTZA1QHX1hyHSmI2Tj1MOoc6_QaHSBJtM",
        },
        {
            name: "Maria Camila Garcia",
            roles: ["pages.about.team.roles.data_scientist"],
            url: "https://www.linkedin.com/in/maria-camila-garc%C3%ADa-salazar-469362241/",
            image: "https://media.licdn.com/dms/image/v2/D4E03AQHkwAssLRrCmQ/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1731373466176?e=1762387200&v=beta&t=No_k4s4CTfi7nCDQAbqy50YF6KfC9-uauX5j4ksf_og",
        },
        {
            name: "Leonardo Gonzalez",
            roles: ["pages.about.team.roles.frontend_developer"],
            url: "https://www.linkedin.com/in/leonardo-gonzalez-castillo-a94098256/",
            image: "https://media.licdn.com/dms/image/v2/D4E03AQG4samTV8NNOg/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1731382900691?e=1762387200&v=beta&t=ytjg6wlxVAP2Ku5Mr-oEWo7lCSFKqKSM8LeLkb_RPzU",
        },
        {
            name: "Michael Taboada",
            roles: ["pages.about.team.roles.data_scientist"],
            url: "https://www.linkedin.com/in/michael-andr%C3%A9s-taboada-naranjo-0263171b1/",
            image: "https://media.licdn.com/dms/image/v2/D5635AQHjKqOpGIlO6A/profile-framedphoto-shrink_200_200/profile-framedphoto-shrink_200_200/0/1713925488966?e=1760295600&v=beta&t=aavL6ghnxDdjr9zDeq0G5FnWqVHHp1Pqyxx70PJvTGE",
        },
        {
            name: "Mauro Gonzalez",
            roles: [
                "pages.about.team.roles.fullstack_developer",
                "pages.about.team.roles.data_scientist",
            ],
            url: "https://www.linkedin.com/in/mauro-gonzalez-figueroa/",
            image: "https://media.licdn.com/dms/image/v2/D4E03AQF-9kGeaEA8zg/profile-displayphoto-scale_200_200/B4EZmMIMO4KQAY-/0/1758992583063?e=1762387200&v=beta&t=bmbw8AAHnhNQYKvNx1fVjuImv2AdOUs2Kn0tXZEPs-I",
        },
    ];

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

        <nav class="relative z-10 p-6 flex justify-between items-center">
            <NuxtLink
                to="/"
                class="inline-flex items-center gap-2 text-white/70 hover:text-white transition-colors"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <path d="m12 19-7-7 7-7" />
                    <path d="M19 12H5" />
                </svg>
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
            >
                <Settings />
            </Button>
        </nav>

        <div class="relative z-10 container mx-auto px-6 py-12 max-w-6xl">
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

            <div class="mb-12">
                <h2 class="text-3xl font-bold mb-8 text-center">
                    {{ $t("pages.about.team.title") }}
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <Card
                        v-for="(member, index) in teamMembers"
                        :key="index"
                        class="bg-white/5 backdrop-blur-sm border-white/10 p-6 hover:bg-white/10 transition-all duration-300 hover:scale-105"
                    >
                        <div class="flex flex-col sm:flex-row gap-6">
                            <div class="flex-shrink-0">
                                <img
                                    :src="member.image"
                                    :alt="member.name"
                                    class="w-32 h-32 rounded-full object-cover border-2 border-white/20"
                                />
                            </div>
                            <div class="flex-1">
                                <h3 class="text-2xl font-bold mb-1 text-white">
                                    {{ member.name }}
                                </h3>
                                <div class="text-blue-400 font-medium mb-3">
                                    <span v-for="(role, idx) in member.roles" :key="role">
                                        {{ $t(role) }}
                                        <span v-if="idx < member.roles.length - 1"> · </span>
                                    </span>
                                </div>
                                <NuxtLink :external="true" :href="member.url">
                                    <ExternalLink />
                                </NuxtLink>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    </main>
</template>

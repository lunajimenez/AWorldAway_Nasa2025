<script setup lang="ts">
    const route = useRoute();
    const head = useLocaleHead();

    const { safeT } = useSafeTranslation();

    const title = computed(() => safeT(String(route.meta.title)));
    const modal = useModal();
</script>

<template>
    <div>
        <Html :lang="head.htmlAttrs.lang" :dir="head.htmlAttrs.dir">
            <Head>
                <Title>{{ title }}</Title>
                <template v-for="link in head.link" :key="link.hid">
                    <Link
                        :id="link.hid"
                        :rel="link.rel"
                        :href="link.href"
                        :hreflang="link.hreflang"
                    />
                </template>
                <template v-for="meta in head.meta" :key="meta.hid">
                    <Meta :id="meta.hid" :property="meta.property" :content="meta.content" />
                </template>
            </Head>
            <Body>
                <Sheet v-if="modal.currentComponent && modal.open" v-model:open="modal.open.value">
                    <component
                        :is="modal.currentComponent.value.value"
                        v-if="modal.currentComponent.value?.value"
                        v-bind="modal.componentProps.value"
                    />
                </Sheet>
                <slot />
            </Body>
        </Html>
    </div>
</template>
